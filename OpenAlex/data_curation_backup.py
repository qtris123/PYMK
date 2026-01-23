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


def collect_stable_concept_ids(keywords: List[str], min_works_count: int = 512004) -> List[Dict[str, Any]]:
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
                
                if 0 <= level <= 2 and works_count > min_works_count:
                    concept_info = {
                        "id": concept.get("id"),
                        "display_name": concept.get("display_name"),
                        "level": level,
                        "works_count": works_count,
                        "keyword": keyword
                    }
                    stable_concepts.append(concept_info)
                    print(f"  ✓ Found: {concept_info['display_name']} (Level {level}, {works_count:,} works)")
            
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


def main():
    """Main function to run the data collection pipeline."""
    random.seed(42)  # For reproducibility
    
    print("\n" + "=" * 80)
    print("OpenAlex Data Curation Pipeline")
    print("=" * 80 + "\n")
    
    # Step 2: Collect stable concept IDs
    concepts = collect_stable_concept_ids(KEYWORDS, min_works_count=512004)
    
    if not concepts:
        print("⚠ Warning: No stable concepts found. Check your filter criteria.")
        return
    
    # Step 3: Retrieve works for each concept
    df_raw, df_filtered = retrieve_works_for_concepts(concepts, per_page=100)
    
    # Display summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nRaw DataFrame shape: {df_raw.shape}")
    print(f"Filtered DataFrame shape: {df_filtered.shape}")
    print(f"\nColumn names (both datasets):")
    for col in df_raw.columns:
        print(f"  - {col}")
    
    # Save both datasets to files (ALWAYS, even if empty)
    raw_output_path = "/home/vo43/PYMK/OpenAlex/collected_works_raw.parquet"
    filtered_output_path = "/home/vo43/PYMK/OpenAlex/collected_works_filtered.parquet"
    df_raw.to_parquet(raw_output_path, index=False)
    df_filtered.to_parquet(filtered_output_path, index=False)
    print(f"\n✓ Raw data saved to: {raw_output_path}")
    print(f"✓ Filtered data saved to: {filtered_output_path}")
    
    # #Citation statistics for FILTERED dataset
    # if not df_filtered.empty:
    #     print(f"\nCitation Statistics (Filtered Dataset):")
    #     print(f"  Mean cited_by_count: {df_filtered['cited_by_count'].mean():.2f}")
    #     print(f"  Median cited_by_count: {df_filtered['cited_by_count'].median():.2f}")
    #     percentile_stats = df_filtered['cited_by_percentile_year'].dropna()
    #     if not percentile_stats.empty:
    #         print(f"  Mean percentile (non-null): {percentile_stats.mean():.2f}")
    #         print(f"  Works with percentile > 0.65: {(percentile_stats > 0.65).sum()}")
        
    #     print(f"\nFirst few rows (Filtered Dataset):")
    #     print(df_filtered[["title", "concept_name", "cited_by_count", "cited_by_percentile_year"]].head(10))
    # else:
    #     print(f"\n⚠ Warning: Filtered dataset is empty!")
    #     print(f"   This likely means recent works don't meet citation criteria.")
    #     print(f"   Consider adjusting filtering thresholds or removing sort by publication_date.")
    
    print("\n" + "=" * 80)



if __name__ == "__main__":
    main()