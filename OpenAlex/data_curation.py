import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import time
import random
from collections import defaultdict


# Step 1: Define keyword set (DONE)
KEYWORDS = [
    "deep learning",
    "representation learning",
    "self-supervised learning",
    "metric learning",
    "optimization algorithms",
    "attention (machine learning)",
    "convolutional neural networks",
    "transformer models",
    "computer vision",
    # "object detection",
    # "image segmentation",
    # "image synthesis",
    # "natural language processing",
    # "language modeling",
    # "sequence-to-sequence learning",
    # "text generation",
    # "multilingual natural language processing",
    # "reinforcement learning",
    # "policy gradient",
    # "value function",
    # "model-based reinforcement learning",
    # "batch reinforcement learning",
    # "large language models",
    # "multimodal machine learning",
    # "sequence modeling",
    # "generative models"
]


def collect_stable_concept_ids(keywords: List[str], min_works_count: int = 51204) -> List[Dict[str, Any]]:
    """
    Step 2: Collect stable concept IDs for each keyword.
    
    Filters by:
    - Level 0-2
    - works_count > min_works_count (default: 512004)
    
    Args:
        keywords: List of keyword strings to search
        min_works_count: Minimum works count threshold
        
    Returns:
        List of concept dictionaries with id, display_name, level, and works_count
    """
    print("=" * 80)
    print("STEP 2: Collecting Stable Concept IDs")
    print("=" * 80)
    
    stable_concepts = []
    
    for keyword in keywords:
        print(f"\nSearching for concept: '{keyword}'")
        
        # Search for concepts matching this keyword
        search_query = keyword.replace(" ", "+")
        url = f"https://api.openalex.org/concepts?search={search_query}&per-page=50"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Filter concepts by level and works_count
            for concept in data.get("results", []):
                level = concept.get("level", -1)
                works_count = concept.get("works_count", 0)
                
                if 0 <= level <= 3 and works_count > min_works_count:
                    concept_info = {
                        "id": concept.get("id"),
                        "display_name": concept.get("display_name"),
                        "level": level,
                        "works_count": works_count,
                        "keyword": keyword
                    }
                    stable_concepts.append(concept_info)
                    print(f"  ✓ Found: {concept_info['display_name']} (Level {level}, {works_count:,} works)")
                else:
                    print(f"  ✗ Skipped: {concept.get('display_name')} (Level {level}, {works_count:,} works)")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ Error searching for '{keyword}': {e}")
            continue
    
    print(f"\n{'=' * 80}")
    print(f"Total stable concepts found: {len(stable_concepts)}")
    print(f"{'=' * 80}\n")
    
    return stable_concepts


def stratified_sample_works(works: List[Dict]) -> List[Dict]:
    """
    Stratified filtering and weighting of works by citation percentile.
    
    Strategy:
    - Keep ALL works with cited_by_percentile_year > 0.65
    - Sample 50% of works with cited_by_percentile_year <= 0.65
    - Keep works with None percentile if cited_by_count > 7
    
    Args:
        works: List of work dictionaries
        
    Returns:
        Filtered and weighted list of works
    """
    # Separate works into categories
    high_percentile = []  # cited_by_percentile_year > 0.65 - KEEP ALL
    low_percentile = []   # cited_by_percentile_year <= 0.65 - SAMPLE 50%
    no_percentile = []    # cited_by_percentile_year is None - KEEP if cited_by_count > 7
    
    for work in works:
        percentile = work.get("cited_by_percentile_year", {})
        if percentile is None:
            percentile_val = None
        else:
            percentile_val = percentile.get("max") if isinstance(percentile, dict) else None
        
        cited_by_count = work.get("cited_by_count", 0)
        
        if percentile_val is not None and percentile_val > 0.65:
            high_percentile.append(work)
        elif percentile_val is not None:
            low_percentile.append(work)
        elif cited_by_count > 7:  # Include if cited_by_count > 7 when percentile is None
            no_percentile.append(work)

        print("High percentile:", len(high_percentile))
        print("Low percentile:", len(low_percentile))
        print("No percentile:", len(no_percentile))
    
    # Apply filtering strategy
    filtered_works = []
    
    # Keep ALL high percentile works
    filtered_works.extend(high_percentile)
    
    # Sample 50% of low percentile works
    if low_percentile:
        n_low = max(1, len(low_percentile) // 2)  # At least 1 if any exist
        filtered_works.extend(random.sample(low_percentile, n_low))
    
    # Keep all works with None percentile but cited_by_count > 7
    filtered_works.extend(no_percentile)
    
    return filtered_works


def retrieve_works_for_concepts(concepts: List[Dict[str, Any]], 
                                per_page: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 3: For each concept, retrieve recent works using concept ID.
    
    Args:
        concepts: List of concept dictionaries from Step 2
        per_page: Number of works to retrieve per API request (OpenAlex limit: 200)
        
    Returns:
        Tuple of (raw_df, filtered_df) - raw contains all works, filtered contains stratified sample
    """
    print("=" * 80)
    print("STEP 3: Retrieving Works for Each Concept")
    print("=" * 80)
    
    all_works_raw = []  # Store ALL works queried (raw dataset)
    all_works_filtered = []  # Store filtered works (filtered dataset)
    raw_count_by_concept = defaultdict(int)
    filtered_count_by_concept = defaultdict(int)
    
    print(f"Works per API request: {per_page}")
    
    for i, concept in enumerate(concepts, 1):
        concept_id = concept["id"]
        concept_name = concept["display_name"]
        
        print(f"[{i}/{len(concepts)}] Processing: {concept_name} ({concept_id})")
        
        try:
            MAX_WORKS = 100 
            results = []
            cursor = "*"

            while len(results) < MAX_WORKS:
                url = (
                    "https://api.openalex.org/works?"
                    f"filter=concepts.id:{concept_id}"
                    "publication_year:>=2018"
                    "&per-page=200"
                    f"&cursor={cursor}"
                )
                data = requests.get(url).json()
                batch = data["results"]

                if not batch:
                    break

                results.extend(batch)
                cursor = data["meta"]["next_cursor"]

                if cursor is None:
                    break

            works = results[:MAX_WORKS]

            if not works:
                print(f"  ✗ No works found")
                continue
            
            # Store ALL works in raw dataset
            for work in works:
                work_data = {
                    "work_id": work.get("id"),
                    "title": work.get("title"),
                    "publication_date": work.get("publication_date"),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "cited_by_percentile_year": work.get("cited_by_percentile_year", {}).get("max") if isinstance(work.get("cited_by_percentile_year"), dict) else None,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "concept_level": concept["level"],
                    "keyword": concept["keyword"],
                    "authorships": work.get("authorships", []),
                    "doi": work.get("doi"),
                    "type": work.get("type"),
                    "open_access": work.get("open_access", {}),
                }
                all_works_raw.append(work_data)
            
            raw_count_by_concept[concept_name] = len(works)
            
            # Apply stratified filtering/weighting for filtered dataset
            # Debug: show citation stats before filtering
            has_percentile = sum(1 for w in works if w.get("cited_by_percentile_year") is not None)
            high_percentile_count = sum(1 for w in works 
                                        if w.get("cited_by_percentile_year", {}) and 
                                        isinstance(w.get("cited_by_percentile_year"), dict) and
                                        w.get("cited_by_percentile_year", {}).get("max", 0) > 0.65)
            avg_citations = sum(w.get("cited_by_count", 0) for w in works) / len(works) if works else 0
            
            print(f"    → Debug: {len(works)} works, {has_percentile} have percentile, {high_percentile_count} > 0.65, avg citations: {avg_citations:.1f}")
            
            filtered_works = stratified_sample_works(works)
            
            # Extract relevant fields and add to filtered collection
            for work in filtered_works:
                work_data = {
                    "work_id": work.get("id"),
                    "title": work.get("title"),
                    "publication_date": work.get("publication_date"),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "cited_by_percentile_year": work.get("cited_by_percentile_year", {}).get("max") if isinstance(work.get("cited_by_percentile_year"), dict) else None,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "concept_level": concept["level"],
                    "keyword": concept["keyword"],
                    "authorships": work.get("authorships", []),
                    "doi": work.get("doi"),
                    "type": work.get("type"),
                    "open_access": work.get("open_access", {}),
                }
                all_works_filtered.append(work_data)
            
            filtered_count_by_concept[concept_name] = len(filtered_works)
            
            print(f"  ✓ Retrieved {len(works)} raw works, {len(filtered_works)} filtered (Total raw: {len(all_works_raw)}, Total filtered: {len(all_works_filtered)})")
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ Error retrieving works: {e}")
            continue
    
    print(f"\n{'=' * 80}")
    print(f"Total raw works collected: {len(all_works_raw)}")
    print(f"Total filtered works collected: {len(all_works_filtered)}")
    print(f"\nRaw works distribution by concept:")
    for concept_name, count in sorted(raw_count_by_concept.items(), key=lambda x: x[1], reverse=True):
        print(f"  {concept_name}: {count}")
    print(f"\nFiltered works distribution by concept:")
    for concept_name, count in sorted(filtered_count_by_concept.items(), key=lambda x: x[1], reverse=True):
        print(f"  {concept_name}: {count}")
    print(f"{'=' * 80}\n")
    
    # Convert to DataFrames
    df_raw = pd.DataFrame(all_works_raw)
    df_filtered = pd.DataFrame(all_works_filtered)
    
    return df_raw, df_filtered


# ============================================================================
# STEP 4: AUTHOR EXTRACTION AND NETWORK BUILDING  
# ============================================================================

def extract_all_authors(df_works):
    """Step 4A: Extract all unique authors from works."""
    print("=" * 80)
    print("STEP 4A: Extracting All Authors")
    print("=" * 80)
    
    author_data = defaultdict(lambda: {'author_name': None, 'institution': None, 'work_ids': [], 'concepts': []})
    
    for _, row in df_works.iterrows():
        authorships = row['authorships']
        work_id = row['work_id']
        concept_id = row['concept_id']
        
        if not isinstance(authorships, list):
            continue
            
        for authorship in authorships:
            author = authorship.get('author', {})
            author_id = author.get('id')
            if not author_id:
                continue
            
            if author_data[author_id]['author_name'] is None:
                author_data[author_id]['author_name'] = author.get('display_name', 'Unknown')
                institutions = authorship.get('institutions', [])
                if institutions:
                    author_data[author_id]['institution'] = institutions[0].get('display_name', 'Unknown')
            
            author_data[author_id]['work_ids'].append(work_id)
            author_data[author_id]['concepts'].append(concept_id)
    
    authors_list = []
    for author_id, data in author_data.items():
        authors_list.append({
            'author_id': author_id,
            'author_name': data['author_name'],
            'institution': data['institution'],
            'work_ids': data['work_ids'],
            'work_count': len(data['work_ids']),
            'concepts': data['concepts']
        })
    
    authors_df = pd.DataFrame(authors_list)
    print(f"\nTotal unique authors: {len(authors_df)}")
    print(f"Average works per author: {authors_df['work_count'].mean():.2f}")
    print(f"{'=' * 80}\n")
    return authors_df


def compute_author_topics(authors_df, df_works):
    """Step 4B: Compute topic distribution and topical purity."""
    print("=" * 80)
    print("STEP 4B: Computing Author Topic Distributions")
    print("=" * 80)
    
    topic_distributions = []
    dominant_topics = []
    
    for _, author in authors_df.iterrows():
        concepts = author['concepts']
        topic_dist = defaultdict(int)
        for concept_id in concepts:
            topic_dist[concept_id] += 1
        
        total = sum(topic_dist.values())
        dominant = max(topic_dist.items(), key=lambda x: x[1])[0] if topic_dist else None
        topic_distributions.append(dict(topic_dist))
        dominant_topics.append(dominant)
    
    authors_df['topic_distribution'] = topic_distributions
    authors_df['dominant_topic'] = dominant_topics
    
    print(f"\nFetching total works from OpenAlex for {len(authors_df)} authors...")
    total_works_list = []
    topical_purity_list = []
    
    for i, row in authors_df.iterrows():
        author_id = row['author_id']
        works_in_dataset = row['work_count']
        
        try:
            response = requests.get(f"https://api.openalex.org/authors/{author_id}")
            response.raise_for_status()
            author_meta = response.json()
            total_works = author_meta.get('works_count', works_in_dataset)
            topical_purity = works_in_dataset / total_works if total_works > 0 else 0.0
            total_works_list.append(total_works)
            topical_purity_list.append(topical_purity)
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(authors_df)} authors...")
            time.sleep(0.05)
        except:
            total_works_list.append(works_in_dataset)
            topical_purity_list.append(1.0)
    
    authors_df['total_works_openalex'] = total_works_list
    authors_df['topical_purity'] = topical_purity_list
    print(f"\n{'=' * 80}")
    print(f"Mean topical purity: {authors_df['topical_purity'].mean():.3f}")
    print(f"{'=' * 80}\n")
    return authors_df


def mark_author_anomalies(authors_df, high_productivity_threshold=10000, low_purity_threshold=0.3):
    """Step 4C: Mark authors with anomalies."""
    print("=" * 80)
    print("STEP 4C: Marking Author Anomalies")
    print("=" * 80)
    authors_df['is_high_productivity'] = authors_df['total_works_openalex'] > high_productivity_threshold
    authors_df['is_low_purity'] = authors_df['topical_purity'] < low_purity_threshold
    print(f"\nHigh productivity: {authors_df['is_high_productivity'].sum()}, Low purity: {authors_df['is_low_purity'].sum()}")
    print(f"{'=' * 80}\n")
    return authors_df


def build_coauthor_edges(df_works, max_degree=50):
    """Step 4D: Build coauthor network."""
    print("=" * 80)
    print("STEP 4D: Building Coauthor Network")
    print("=" * 80)
    edges = defaultdict(list)
    
    for _, row in df_works.iterrows():
        authorships = row['authorships']
        work_id = row['work_id']
        if not isinstance(authorships, list) or len(authorships) < 2:
            continue
        author_ids = [a.get('author', {}).get('id') for a in authorships if a.get('author', {}).get('id')]
        
        for i in range(len(author_ids)):
            for j in range(i + 1, len(author_ids)):
                edge = tuple(sorted([author_ids[i], author_ids[j]]))
                edges[edge].append(work_id)
    
    edge_list = [{'author_id_1': a1, 'author_id_2': a2, 'weight': len(wids), 'work_ids': wids} 
                 for (a1, a2), wids in edges.items()]
    edges_df = pd.DataFrame(edge_list)
    
    if max_degree > 0 and len(edges_df) > 0:
        print(f"Applying max_degree={max_degree}...")
        filtered = []
        counts = defaultdict(int)
        for _, edge in edges_df.sort_values('weight', ascending=False).iterrows():
            a1, a2 = edge['author_id_1'], edge['author_id_2']
            if counts[a1] < max_degree and counts[a2] < max_degree:
                filtered.append(edge)
                counts[a1] += 1
                counts[a2] += 1
        edges_df = pd.DataFrame(filtered)
    
    print(f"\nTotal coauthor edges: {len(edges_df)}")
    print(f"{'=' * 80}\n")
    return edges_df


def build_citation_edges(df_works, authors_df):
    """Step 4E: Build citation network."""
    print("=" * 80)
    print("STEP 4E: Building Citation Network")
    print("=" * 80)
    
    work_to_authors = {}
    for _, row in df_works.iterrows():
        work_id = row['work_id']
        authorships = row['authorships']
        if isinstance(authorships, list):
            work_to_authors[work_id] = [a.get('author', {}).get('id') for a in authorships if a.get('author', {}).get('id')]
    
    citation_edges = defaultdict(lambda: {'weight': 0, 'work_pairs': []})
    
    for _, row in df_works.iterrows():
        citing_work = row['work_id']
        citing_authors = work_to_authors.get(citing_work, [])
        try:
            resp = requests.get(f"https://api.openalex.org/works/{citing_work}")
            resp.raise_for_status()
            referenced = resp.json().get('referenced_works', [])
            
            for cited_work in referenced:
                if cited_work in work_to_authors:
                    for ca in citing_authors:
                        for cd in work_to_authors[cited_work]:
                            if ca != cd:
                                citation_edges[(ca, cd)]['weight'] += 1
                                citation_edges[(ca, cd)]['work_pairs'].append((citing_work, cited_work))
            time.sleep(0.05)
        except:
            continue
    
    edge_list = [{'citing_author_id': ca, 'cited_author_id': cd, 'weight': d['weight'], 'work_pairs': d['work_pairs']} 
                 for (ca, cd), d in citation_edges.items()]
    citation_df = pd.DataFrame(edge_list)
    print(f"\nTotal citation edges: {len(citation_df)}")
    print(f"{'=' * 80}\n")
    return citation_df


def evaluate_author_network(authors_df, coauthor_edges_df, citation_edges_df):
    """Step 4F: Evaluate network."""
    print("=" * 80)
    print("STEP 4F: NETWORK EVALUATION")
    print("=" * 80)
    print(f"\nAuthors: {len(authors_df)}, Mean works/author: {authors_df['work_count'].mean():.2f}")
    print(f"Mean topical purity: {authors_df['topical_purity'].mean():.3f}")
    print(f"Coauthor edges: {len(coauthor_edges_df)}, Citation edges: {len(citation_edges_df)}")
    print(f"{'=' * 80}\n")


def main():
    """Main pipeline."""
    random.seed(42)
    print("\n" + "=" * 80)
    print("OpenAlex Data Curation Pipeline")
    print("=" * 80 + "\n")
    
    concepts = collect_stable_concept_ids(KEYWORDS, min_works_count=512004)
    if not concepts:
        print("⚠ No concepts found")
        return
    if open("/home/vo43/PYMK/OpenAlex/collected_works_raw.parquet"):
        df_raw = pd.read_parquet("/home/vo43/PYMK/OpenAlex/collected_works_raw.parquet")
        df_filtered = pd.read_parquet("/home/vo43/PYMK/OpenAlex/collected_works_filtered.parquet")
        print(f"\n✓ Loaded: /collected_works_raw.parquet ({df_raw.shape[0]} works)")
        print(f"✓ Loaded: /collected_works_filtered.parquet ({df_filtered.shape[0]} works)\n")
    else:
        df_raw, df_filtered = retrieve_works_for_concepts(concepts, per_page=100)
        raw_out = "/home/vo43/PYMK/OpenAlex/collected_works_raw.parquet"
        filt_out = "/home/vo43/PYMK/OpenAlex/collected_works_filtered.parquet"
        df_raw.to_parquet(raw_out, index=False)
        df_filtered.to_parquet(filt_out, index=False)
        print(f"\n✓ Saved: {raw_out} ({df_raw.shape[0]} works)")
        print(f"✓ Saved: {filt_out} ({df_filtered.shape[0]} works)\n")
    
    if df_raw.empty:
        return
    
    print("=" * 80)
    print("STEP 4: AUTHOR NETWORK")
    print("=" * 80 + "\n")
    
    authors_df = extract_all_authors(df_filtered)
    authors_df = compute_author_topics(authors_df, df_filtered)
    authors_df = mark_author_anomalies(authors_df)
    coauthor_edges = build_coauthor_edges(df_filtered, max_degree=50)
    citation_edges = build_citation_edges(df_filtered, authors_df)
    
    authors_df.to_parquet("/home/vo43/PYMK/OpenAlex/authors.parquet", index=False)
    coauthor_edges.to_parquet("/home/vo43/PYMK/OpenAlex/coauthor_edges.parquet", index=False)
    citation_edges.to_parquet("/home/vo43/PYMK/OpenAlex/citation_edges.parquet", index=False)
    
    evaluate_author_network(authors_df, coauthor_edges, citation_edges)
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
