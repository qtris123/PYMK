# Graph-Based Author Candidate Generation

A temporal graph-based recommendation system for "People You May Know" (PYMK) style author recommendations using OpenAlex academic data.

## Overview

This project implements a non-GNN graph-based candidate generation algorithm that:
- Splits collaboration data by time for realistic evaluation
- Extracts N-hop neighbors (2-hop and 3-hop) as candidates
- Ranks candidates using common neighbor counting
- Evaluates predictions against actual future collaborations

## Features

✅ **Temporal Train/Test Splitting** - Prevents data leakage by splitting edges based on publication year  
✅ **N-hop Neighbor Extraction** - Efficiently computes multi-hop neighborhoods using BFS  
✅ **Common Neighbor Ranking** - Ranks candidates by shared connections with the target user  
✅ **Automatic Evaluation** - Validates recommendations against actual test collaborations  
✅ **Rich Author Metadata** - Includes institution, topics, and publication statistics

## Quick Start

### 1. Prepare the Data

```bash
python graph_data_prep.py --cutoff-year 2019 --max-hops 3 --output-dir ./split_data
```

**Output:**
```
Loaded 903 authors
Loaded 300 works
Loaded 2983 coauthor edges

Train/Test Split (cutoff year: 2019):
  Train coauthor edges: 857
  Test coauthor edges: 2126

N-hop neighbors computed up to 3 hops
```

### 2. Generate Recommendations

```bash
# Run demo with random user
python candidate_generation_example.py --demo

# Generate for specific user
python candidate_generation_example.py --user-id "https://openalex.org/A5100748135"
```

## Example Output

```
================================================================================
CANDIDATE RECOMMENDATIONS FOR USER: https://openalex.org/A5100748135
================================================================================

User: Yu Qiao
Institution: Shenzhen Institutes of Advanced Technology
Dominant Topic: https://openalex.org/C108583219
Work Count: 3

Top 100 Candidates (ranked by common neighbors):
--------------------------------------------------------------------------------
Rank   Hop    Common     Name                           Topic               
--------------------------------------------------------------------------------
1      2      1          Zhirong Wu                     https://openalex.o  
2      2      1          Ziwei Liu                      https://openalex.o  
3      2      1          Chen Change Loy                https://openalex.o  
4      2      1          John Winn                      https://openalex.o  
5      2      1          Herbert Bay                    https://openalex.o  
6      2      1          Andrew Zisserman               https://openalex.o  
7      2      1          Aditya Khosla                  https://openalex.o  
8      2      1          Andreas Ess                    https://openalex.o  
9      2      1          Chao Dong                      https://openalex.o  
10     2      1          Jianxiong Xiao                 https://openalex.o  
...
18     2      1          Kaiming He                     https://openalex.o  
19     2      1          Linguang Zhang                 https://openalex.o  
20     3      0          Shaoqing Ren                   https://openalex.o  
...
--------------------------------------------------------------------------------

Evaluation (against test set):
  Total candidates recommended: 51
  Actual new collaborators in test: 7
  Hits (correct predictions): 2
  Hit rate: 3.92%
  Recall: 28.57%

  Sample correct predictions:
    - Chen Change Loy (https://openalex.org/C108583219)
    - Chao Dong (https://openalex.org/C108583219)

================================================================================

Candidate Distribution by Hop:
  2-hop: 19 candidates
  3-hop: 32 candidates

Common Neighbor Statistics:
  Min: 0
  Max: 1
  Mean: 0.37
  Median: 0
```

## How It Works

### 1. Temporal Graph Construction

```
Timeline:  |------------ TRAIN ------------|---- TEST ----|
           1944                           2019          2025
                                           ↑
                                      cutoff year
```

- **Train Graph**: All collaborations before 2019
- **Test Graph**: New collaborations from 2019 onwards
- **No Leakage**: Test edges never appear in training data

### 2. N-hop Neighbor Extraction

```
User → 1-hop → 2-hop → 3-hop
       (direct)  ↑       ↑
              candidates candidates
```

- Uses BFS to find neighbors at distance 2 and 3
- Excludes existing 1-hop neighbors (already connected)
- Pre-computes for all authors for efficiency

### 3. Common Neighbor Ranking

For each candidate, compute:

```
score = |neighbors(user) ∩ neighbors(candidate)|
```

Sort candidates by:
1. Common neighbor count (descending)
2. Hop distance (ascending)

### 4. Evaluation

Predictions are validated against actual test collaborations:
- **Hit Rate**: Percentage of recommendations that became actual collaborators
- **Recall**: Percentage of actual collaborators that were recommended

## Data Format

### Input Files
- `authors.parquet` - Author metadata (903 authors)
- `collected_works_filtered.parquet` - Publications with dates (300 works)
- `coauthor_edges.parquet` - Collaboration edges (2,983 edges)

### Output Structure

```
split_data/
├── authors.parquet                    # All author metadata
├── metadata.json                       # Split statistics
├── train/
│   ├── coauthor_edges.parquet         # Training edges with years
│   ├── coauthor_adjlist.json          # Graph adjacency list
│   └── n_hop_neighbors.json           # Pre-computed N-hop neighborhoods
└── test/
    ├── coauthor_edges.parquet         # Test edges with years
    ├── coauthor_adjlist.json          # Full graph (train + test)
    └── n_hop_neighbors.json           # N-hop neighbors for evaluation
```

## Configuration

### Data Preparation Options

```bash
python graph_data_prep.py \
  --data-dir /path/to/data \      # Input data directory
  --cutoff-year 2019 \            # Train/test split year
  --max-hops 3 \                  # Maximum hop distance
  --output-dir ./split_data       # Output directory
```

### Candidate Generation Options

```bash
python candidate_generation_example.py \
  --user-id "..." \               # Specific user ID
  --hops 2 3 \                    # Hop distances to consider
  --max-candidates 100 \          # Maximum candidates to return
  --top-k 100 \                   # Top K to display
  --demo                          # Run with random user
```

## Algorithm Performance

Based on demo results:
- **Average Hit Rate**: ~3-4% (candidates that became actual collaborators)
- **Average Recall**: ~28% (actual collaborators that were predicted)
- **Candidate Distribution**: ~37% at 2-hop, ~63% at 3-hop
- **Common Neighbors**: Mean of 0.37, indicating sparse connectivity

## Use Cases

This pipeline supports various graph-based algorithms:
- **Common Neighbors**: Simple count of shared connections
- **Jaccard Coefficient**: Normalized similarity measure
- **Adamic-Adar Index**: Weighted by inverse neighbor degree
- **Resource Allocation**: Weighted by neighbor inverse degree
- **Preferential Attachment**: Product of node degrees

## Technical Details

### Temporal Leakage Prevention
- Edges are timestamped using publication dates
- Training edges: year < cutoff_year
- Test edges: year >= cutoff_year
- Zero overlap guaranteed between train/test

### Graph Representation
- **Undirected**: Coauthor relationships are symmetric
- **Weighted**: Edge weights represent collaboration frequency
- **Temporal**: Each edge has first collaboration year

### Efficiency
- Pre-computed N-hop neighbors for all authors
- JSON adjacency lists for fast lookups
- Parquet format for efficient data storage

## Dependencies

```bash
pip install pandas numpy pyarrow
```

## Dataset Statistics

```json
{
  "cutoff_year": 2019,
  "max_hops": 3,
  "statistics": {
    "total_authors": 903,
    "total_works": 300,
    "train": {
      "coauthor_edges": 857,
      "unique_authors_in_coauthor_network": 740,
      "year_range": "1964-2018"
    },
    "test": {
      "coauthor_edges": 2126,
      "unique_authors_in_coauthor_network": 870,
      "year_range": "2019-2025"
    }
  }
}
```

## Files

- `graph_data_prep.py` - Main data processing pipeline (~350 lines)
- `candidate_generation_example.py` - Example usage and evaluation (~250 lines)
- `data_curation.py` - Data collection and preprocessing

## Future Enhancements

- [ ] Implement additional ranking algorithms (Adamic-Adar, Jaccard)
- [ ] Add content-based filtering using research topics
- [ ] Support for multiple evaluation metrics (MRR, MAP, NDCG)
- [ ] Batch recommendation generation for all users
- [ ] Interactive visualization of recommendation results

## License

This project uses data from [OpenAlex](https://openalex.org/), which is available under CC0 license.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
