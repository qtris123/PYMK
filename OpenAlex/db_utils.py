"""
db_utils.py
===========
DuckDB schema and upsert helpers for the OpenAlex pipeline.
All raw SQL lives here; no pipeline logic.
"""

import json
import duckdb
from typing import List, Set

# ---------------------------------------------------------------------------
# Country → Continent (ISO 3166-1 alpha-2)
# ---------------------------------------------------------------------------

COUNTRY_TO_CONTINENT = {
    # Africa
    "DZ":"Africa","AO":"Africa","BJ":"Africa","BW":"Africa","BF":"Africa","BI":"Africa",
    "CM":"Africa","CV":"Africa","CF":"Africa","TD":"Africa","KM":"Africa","CD":"Africa",
    "CG":"Africa","CI":"Africa","DJ":"Africa","EG":"Africa","GQ":"Africa","ER":"Africa",
    "ET":"Africa","GA":"Africa","GM":"Africa","GH":"Africa","GN":"Africa","GW":"Africa",
    "KE":"Africa","LS":"Africa","LR":"Africa","LY":"Africa","MG":"Africa","MW":"Africa",
    "ML":"Africa","MR":"Africa","MU":"Africa","MA":"Africa","MZ":"Africa","NA":"Africa",
    "NE":"Africa","NG":"Africa","RW":"Africa","ST":"Africa","SN":"Africa","SC":"Africa",
    "SL":"Africa","SO":"Africa","ZA":"Africa","SS":"Africa","SD":"Africa","SZ":"Africa",
    "TZ":"Africa","TG":"Africa","TN":"Africa","UG":"Africa","ZM":"Africa","ZW":"Africa",
    # Americas
    "AG":"Americas","AR":"Americas","BS":"Americas","BB":"Americas","BZ":"Americas",
    "BO":"Americas","BR":"Americas","CA":"Americas","CL":"Americas","CO":"Americas",
    "CR":"Americas","CU":"Americas","DM":"Americas","DO":"Americas","EC":"Americas",
    "SV":"Americas","GD":"Americas","GT":"Americas","GY":"Americas","HT":"Americas",
    "HN":"Americas","JM":"Americas","MX":"Americas","NI":"Americas","PA":"Americas",
    "PY":"Americas","PE":"Americas","KN":"Americas","LC":"Americas","VC":"Americas",
    "SR":"Americas","TT":"Americas","US":"Americas","UY":"Americas","VE":"Americas",
    # Asia
    "AF":"Asia","AM":"Asia","AZ":"Asia","BH":"Asia","BD":"Asia","BT":"Asia","BN":"Asia",
    "KH":"Asia","CN":"Asia","CY":"Asia","GE":"Asia","IN":"Asia","ID":"Asia","IR":"Asia",
    "IQ":"Asia","IL":"Asia","JP":"Asia","JO":"Asia","KZ":"Asia","KW":"Asia","KG":"Asia",
    "LA":"Asia","LB":"Asia","MY":"Asia","MV":"Asia","MN":"Asia","MM":"Asia","NP":"Asia",
    "KP":"Asia","OM":"Asia","PK":"Asia","PS":"Asia","PH":"Asia","QA":"Asia","SA":"Asia",
    "SG":"Asia","KR":"Asia","LK":"Asia","SY":"Asia","TW":"Asia","TJ":"Asia","TH":"Asia",
    "TL":"Asia","TR":"Asia","TM":"Asia","AE":"Asia","UZ":"Asia","VN":"Asia","YE":"Asia",
    # Europe
    "AL":"Europe","AD":"Europe","AT":"Europe","BY":"Europe","BE":"Europe","BA":"Europe",
    "BG":"Europe","HR":"Europe","CZ":"Europe","DK":"Europe","EE":"Europe","FI":"Europe",
    "FR":"Europe","DE":"Europe","GR":"Europe","HU":"Europe","IS":"Europe","IE":"Europe",
    "IT":"Europe","LV":"Europe","LI":"Europe","LT":"Europe","LU":"Europe","MT":"Europe",
    "MD":"Europe","MC":"Europe","ME":"Europe","NL":"Europe","MK":"Europe","NO":"Europe",
    "PL":"Europe","PT":"Europe","RO":"Europe","RU":"Europe","SM":"Europe","RS":"Europe",
    "SK":"Europe","SI":"Europe","ES":"Europe","SE":"Europe","CH":"Europe","UA":"Europe",
    "GB":"Europe","VA":"Europe","XK":"Europe",
    # Oceania
    "AU":"Oceania","FJ":"Oceania","KI":"Oceania","MH":"Oceania","FM":"Oceania",
    "NR":"Oceania","NZ":"Oceania","PW":"Oceania","PG":"Oceania","WS":"Oceania",
    "SB":"Oceania","TO":"Oceania","TV":"Oceania","VU":"Oceania",
}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db(con: duckdb.DuckDBPyConnection):
    """Create all tables if they don't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS works (
            work_id          TEXT PRIMARY KEY,
            title            TEXT,
            publication_date TEXT,
            publication_year INT,
            type             TEXT,
            is_oa            BOOLEAN,
            concepts         JSON,
            authorships      JSON,
            referenced_works JSON,
            keyword          TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS authors (
            author_id                   TEXT PRIMARY KEY,
            author_name                 TEXT,
            total_works_openalex        INT,
            total_cited_by              INT,
            h_index                     INT,
            i10_index                   INT,
            two_yr_mean_citedness       FLOAT,
            counts_by_year              JSON,
            affiliations_json           JSON,
            last_known_country          TEXT,
            last_known_institution_name TEXT,
            last_known_institution_type TEXT,
            enriched                    BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS author_works (
            author_id TEXT,
            work_id   TEXT,
            PRIMARY KEY (author_id, work_id)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS author_temporal_features (
            author_id        TEXT,
            year             INT,
            country_code_t   TEXT,
            continent_t      TEXT,
            works_count_t    INT,
            cited_by_count_t INT,
            oa_works_count_t INT,
            cum_works_t      INT,
            cum_citations_t  INT,
            PRIMARY KEY (author_id, year)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS coauthor_edges (
            author_id_1 TEXT,
            author_id_2 TEXT,
            weight      INT,
            PRIMARY KEY (author_id_1, author_id_2)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS citation_edges (
            citing_author_id TEXT,
            cited_author_id  TEXT,
            weight           INT,
            PRIMARY KEY (citing_author_id, cited_author_id)
        )
    """)


# ---------------------------------------------------------------------------
# Works
# ---------------------------------------------------------------------------

def get_existing_work_ids(con) -> Set[str]:
    return {r[0] for r in con.execute("SELECT work_id FROM works").fetchall()}


def upsert_works(con, rows: List[dict]):
    if not rows:
        return
    vals = [
        (
            r["work_id"], r.get("title"), r.get("publication_date"),
            r.get("publication_year"), r.get("type"), r.get("is_oa"),
            json.dumps(r.get("concepts", [])),
            json.dumps(r.get("authorships", [])),
            json.dumps(r.get("referenced_works", [])),
            r.get("keyword"),
        )
        for r in rows
    ]
    con.executemany("INSERT OR REPLACE INTO works VALUES (?,?,?,?,?,?,?,?,?,?)", vals)


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------

def upsert_authors_stub(con, rows: List[dict]):
    """Insert author_id + author_name only; skip if already exists."""
    if not rows:
        return
    con.executemany(
        "INSERT INTO authors(author_id, author_name) VALUES (?,?) ON CONFLICT DO NOTHING",
        [(r["author_id"], r["author_name"]) for r in rows],
    )


def upsert_author_works(con, pairs: List[tuple]):
    """pairs: list of (author_id, work_id)"""
    if not pairs:
        return
    con.executemany(
        "INSERT INTO author_works VALUES (?,?) ON CONFLICT DO NOTHING", pairs
    )


def get_unenriched_author_ids(con) -> List[str]:
    return [
        r[0] for r in con.execute(
            "SELECT author_id FROM authors WHERE NOT enriched"
        ).fetchall()
    ]


def update_author_enrichment(con, rows: List[dict]):
    """UPDATE existing author rows with data fetched from the /authors API."""
    if not rows:
        return
    vals = [
        (
            r["total_works_openalex"], r["total_cited_by"],
            r["h_index"], r["i10_index"], r["two_yr_mean_citedness"],
            json.dumps(r["counts_by_year"]),
            json.dumps(r["affiliations_json"]),
            r["last_known_country"],
            r["last_known_institution_name"],
            r["last_known_institution_type"],
            r["author_id"],  # WHERE clause
        )
        for r in rows
    ]
    con.executemany("""
        UPDATE authors SET
            total_works_openalex=?, total_cited_by=?,
            h_index=?, i10_index=?, two_yr_mean_citedness=?,
            counts_by_year=?, affiliations_json=?,
            last_known_country=?, last_known_institution_name=?,
            last_known_institution_type=?, enriched=TRUE
        WHERE author_id=?
    """, vals)


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def upsert_temporal_features(con, rows: List[dict]):
    if not rows:
        return
    vals = [
        (
            r["author_id"], r["year"], r["country_code_t"], r["continent_t"],
            r["works_count_t"], r["cited_by_count_t"], r["oa_works_count_t"],
            r["cum_works_t"], r["cum_citations_t"],
        )
        for r in rows
    ]
    con.executemany(
        "INSERT OR REPLACE INTO author_temporal_features VALUES (?,?,?,?,?,?,?,?,?)", vals
    )


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def compute_and_upsert_coauthor_edges(con, max_degree: int = 50):
    """
    Computes coauthor weights and applies a degree cap using SQL.
    This is significantly faster than Python-side loops.
    """
    # 1. Clear old edges (optional depending on upsert needs,
    # but for a "recompute all" stage it's cleaner)
    con.execute("DELETE FROM coauthor_edges")

    # 2. Compute weights and insert top edges per author
    # Note: We use a window function to implement a degree cap similar
    # to the Python logic (top N neighbors by weight).
    con.execute(f"""
        INSERT INTO coauthor_edges
        WITH pairs AS (
            SELECT
                LEAST(a1.author_id, a2.author_id)    AS aid1,
                GREATEST(a1.author_id, a2.author_id)  AS aid2
            FROM author_works a1
            JOIN author_works a2
              ON a1.work_id = a2.work_id AND a1.author_id < a2.author_id
        ),
        weighted AS (
            SELECT aid1, aid2, COUNT(*) AS weight
            FROM pairs
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT aid1, aid2, weight,
                   row_number() OVER (PARTITION BY aid1 ORDER BY weight DESC) as r1,
                   row_number() OVER (PARTITION BY aid2 ORDER BY weight DESC) as r2
            FROM weighted
        )
        SELECT aid1, aid2, weight
        FROM ranked
        WHERE r1 <= {max_degree} AND r2 <= {max_degree}
    """)


def compute_and_upsert_citation_edges(con):
    """
    Computes citation weights between authors using SQL.
    Unnests JSON referenced_works and joins back to author_works.
    """
    con.execute("DELETE FROM citation_edges")

    con.execute("""
        INSERT INTO citation_edges
        WITH refs AS (
            SELECT
                aw.author_id as citing_aid,
                unnest(cast(json_extract(w.referenced_works, '$[*]') as TEXT[])) as cited_wid
            FROM works w
            JOIN author_works aw ON w.work_id = aw.work_id
        ),
        edges AS (
            SELECT
                r.citing_aid,
                aw_cited.author_id as cited_aid
            FROM refs r
            JOIN author_works aw_cited ON r.cited_wid = aw_cited.work_id
            WHERE r.citing_aid != aw_cited.author_id
        )
        SELECT citing_aid, cited_aid, COUNT(*) as weight
        FROM edges
        GROUP BY 1, 2
    """)
