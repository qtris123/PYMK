"""
data_curation_db.py
====================
OpenAlex data collection pipeline — DuckDB-backed.

Steps:
  1. collect_all_works      — fetch works by keyword, upsert (incremental)
  2. extract_authors        — pull author stubs from works, upsert
  3. fetch_author_objects   — batch-fetch full /authors records, upsert
  4. build_temporal_features— explode counts_by_year + affiliations → per-year rows
  5. build_coauthor_edges   — coauthor pairs from shared works
  6. build_citation_edges   — citation pairs from referenced_works

All DB helpers and schema are in db_utils.py.
"""

import os
import json
import time
import requests
import duckdb
from collections import defaultdict
from typing import List

from db_utils import (
    COUNTRY_TO_CONTINENT,
    init_db,
    get_existing_work_ids, upsert_works,
    upsert_authors_stub, upsert_author_works,
    get_unenriched_author_ids, update_author_enrichment,
    upsert_temporal_features,
    compute_and_upsert_coauthor_edges, compute_and_upsert_citation_edges,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KEYWORDS = [
    "deep learning", "supervised learning", "unsupervised learning",
    # "representation learning", "self-supervised learning", "metric learning",
    # "optimization algorithms", "attention mechanisms",
    # "convolutional neural networks", "transformer models",
    # "large language models", "graph neural networks",
    # "efficient training and inference",
]

MAILTO              = "vo43@purdue.edu"
BASE_URL            = "https://api.openalex.org"
DB_PATH             = os.path.join(os.path.dirname(__file__), "openalex.duckdb")
MAX_WORKS_PER_KW    = 50
MIN_PUBLICATION_YEAR = 2018


# ---------------------------------------------------------------------------
# Step 1: Collect works
# ---------------------------------------------------------------------------

def collect_all_works(con: duckdb.DuckDBPyConnection):
    print("=" * 60)
    print("STEP 1: Collecting Works")
    print("=" * 60)

    existing = get_existing_work_ids(con)
    print(f"  Already in DB: {len(existing)}")

    total_new = 0
    for keyword in KEYWORDS:
        batch = _fetch_works_for_keyword(keyword, existing)
        if batch:
            upsert_works(con, batch)
            existing.update(w["work_id"] for w in batch)
            total_new += len(batch)
        print(f"  '{keyword}': +{len(batch)} new")
        time.sleep(0.1)

    print(f"  Done. {total_new} new works added.\n")


def _fetch_works_for_keyword(keyword: str, skip_ids: set) -> List[dict]:
    results = []
    cursor = "*"
    while len(results) < MAX_WORKS_PER_KW:
        url = (
            f"{BASE_URL}/works?"
            f"search={keyword}"
            f"&filter=publication_year:>{MIN_PUBLICATION_YEAR}"
            f"&select=id,title,authorships,referenced_works,publication_date,type,open_access,concepts"
            f"&sort=cited_by_count:desc"
            f"&per_page={MAX_WORKS_PER_KW}"
            f"&cursor={cursor}"
            f"&mailto={MAILTO}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break

        for w in batch:
            wid = w.get("id")
            if not wid or wid in skip_ids:
                continue
            pub_date = w.get("publication_date", "") or ""
            oa = w.get("open_access") or {}
            results.append({
                "work_id":          wid,
                "title":            w.get("title"),
                "publication_date": pub_date,
                "publication_year": int(pub_date[:4]) if len(pub_date) >= 4 else None,
                "type":             w.get("type"),
                "is_oa":            oa.get("is_oa", False),
                "concepts":         [
                    {"id": c["id"], "level": c.get("level"), "score": c.get("score")}
                    for c in w.get("concepts", [])
                ],
                "authorships":      w.get("authorships", []),
                "referenced_works": w.get("referenced_works", []),
                "keyword":          keyword,
            })

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return results[:MAX_WORKS_PER_KW]


# ---------------------------------------------------------------------------
# Step 2: Extract authors from works
# ---------------------------------------------------------------------------

def extract_authors(con: duckdb.DuckDBPyConnection):
    print("=" * 60)
    print("STEP 2: Extracting Authors from Works")
    print("=" * 60)

    rows = con.execute("SELECT work_id, authorships FROM works").fetchall()

    author_stubs = {}      # author_id -> {author_id, author_name}
    author_work_pairs = [] # (author_id, work_id)

    for work_id, authorships_raw in rows:
        authorships = _parse_json(authorships_raw)
        for authorship in authorships:
            author = authorship.get("author") or {}
            aid = author.get("id")
            if not aid:
                continue
            if aid not in author_stubs:
                author_stubs[aid] = {
                    "author_id":   aid,
                    "author_name": author.get("display_name", "Unknown"),
                }
            author_work_pairs.append((aid, work_id))

    upsert_authors_stub(con, list(author_stubs.values()))
    upsert_author_works(con, author_work_pairs)
    print(f"  {len(author_stubs)} unique authors, {len(author_work_pairs)} author-work pairs\n")


# ---------------------------------------------------------------------------
# Step 3: Fetch full author objects from /authors API
# ---------------------------------------------------------------------------

def fetch_author_objects(con: duckdb.DuckDBPyConnection):
    print("=" * 60)
    print("STEP 3: Fetching Author Objects")
    print("=" * 60)

    author_ids = get_unenriched_author_ids(con)
    print(f"  To enrich: {len(author_ids)}")
    if not author_ids:
        print("  All authors already enriched.\n")
        return

    batches = [author_ids[i:i + 50] for i in range(0, len(author_ids), 50)]
    enriched_count = 0

    for i, batch in enumerate(batches, 1):
        short_ids = "|".join(aid.replace("https://openalex.org/", "") for aid in batch)
        url = (
            f"{BASE_URL}/authors?"
            f"filter=ids.openalex:{short_ids}"
            f"&select=id,works_count,cited_by_count,summary_stats,counts_by_year,affiliations,last_known_institutions"
            f"&per_page=50"
            f"&mailto={MAILTO}"
        )
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            enrichment_rows = []
            for a in resp.json().get("results", []):
                stats = a.get("summary_stats") or {}
                lki   = ((a.get("last_known_institutions") or []) + [{}])[0]
                enrichment_rows.append({
                    "author_id":                   a["id"],
                    "total_works_openalex":         a.get("works_count", 0),
                    "total_cited_by":               a.get("cited_by_count", 0),
                    "h_index":                      stats.get("h_index", 0),
                    "i10_index":                    stats.get("i10_index", 0),
                    "two_yr_mean_citedness":        stats.get("2yr_mean_citedness", 0.0),
                    "counts_by_year":               a.get("counts_by_year", []),
                    "affiliations_json":            a.get("affiliations", []),
                    "last_known_country":           lki.get("country_code"),
                    "last_known_institution_name":  lki.get("display_name"),
                    "last_known_institution_type":  lki.get("type"),
                })
            update_author_enrichment(con, enrichment_rows)
            enriched_count += len(enrichment_rows)
        except Exception as e:
            print(f"  Warning: batch {i} failed: {e}")

        if i % 20 == 0 or i == len(batches):
            print(f"  Batch {i}/{len(batches)} done")
        time.sleep(0.05)

    print(f"  Enriched {enriched_count}/{len(author_ids)} authors\n")


# ---------------------------------------------------------------------------
# Step 4: Build temporal features
# ---------------------------------------------------------------------------

def build_temporal_features(con: duckdb.DuckDBPyConnection):
    print("=" * 60)
    print("STEP 4: Building Temporal Features")
    print("=" * 60)

    rows = con.execute(
        "SELECT author_id, counts_by_year, affiliations_json FROM authors WHERE enriched"
    ).fetchall()

    feature_rows = []
    for author_id, cby_raw, aff_raw in rows:
        counts_by_year = _parse_json(cby_raw)
        affiliations   = _parse_json(aff_raw)

        # year → country_code from affiliation history
        year_to_country = {}
        for aff in affiliations:
            inst    = aff.get("institution") or {}
            country = inst.get("country_code")
            for yr in aff.get("years", []):
                year_to_country[yr] = country

        # sort ascending, accumulate
        cum_works = cum_cites = 0
        for entry in sorted(counts_by_year, key=lambda x: x["year"]):
            yr  = entry["year"]
            wc  = entry.get("works_count", 0)
            cc  = entry.get("cited_by_count", 0)
            oa  = entry.get("oa_works_count", 0)
            cum_works += wc
            cum_cites += cc
            country   = year_to_country.get(yr)
            feature_rows.append({
                "author_id":        author_id,
                "year":             yr,
                "country_code_t":   country,
                "continent_t":      COUNTRY_TO_CONTINENT.get(country) if country else None,
                "works_count_t":    wc,
                "cited_by_count_t": cc,
                "oa_works_count_t": oa,
                "cum_works_t":      cum_works,
                "cum_citations_t":  cum_cites,
            })

    upsert_temporal_features(con, feature_rows)
    print(f"  {len(feature_rows)} rows for {len(rows)} authors\n")


# # ---------------------------------------------------------------------------
# # Step 5: Build coauthor edges
# # ---------------------------------------------------------------------------

# def build_coauthor_edges(con: duckdb.DuckDBPyConnection, max_degree: int = 50):
#     print("=" * 60)
#     print("STEP 5: Building Coauthor Edges (SQL Optimized)")
#     print("=" * 60)
#     compute_and_upsert_coauthor_edges(con, max_degree)
#     count = con.execute("SELECT COUNT(*) FROM coauthor_edges").fetchone()[0]
#     print(f"  {count} coauthor edges computed.\n")


# # ---------------------------------------------------------------------------
# # Step 6: Build citation edges
# # ---------------------------------------------------------------------------

# def build_citation_edges(con: duckdb.DuckDBPyConnection):
#     print("=" * 60)
#     print("STEP 6: Building Citation Edges (SQL Optimized)")
#     print("=" * 60)
#     compute_and_upsert_citation_edges(con)
#     count = con.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0]
#     print(f"  {count} citation edges computed.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(value):
    """DuckDB may return JSON columns as strings or already-parsed objects."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("OpenAlex DuckDB Pipeline")
    print("=" * 60 + "\n")

    con = duckdb.connect(DB_PATH)
    init_db(con)

    collect_all_works(con)
    extract_authors(con)
    fetch_author_objects(con)
    build_temporal_features(con)
    # build_coauthor_edges(con) # consider report node March 11th
    # build_citation_edges(con)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for table in ["works", "authors", "author_works",
                  "author_temporal_features", "coauthor_edges", "citation_edges"]:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<30} {n:>8} rows")
    print("=" * 60 + "\n")

    con.close()


if __name__ == "__main__":
    main()
