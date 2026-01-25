"""
Graph Data Preparation for Candidate Generation
=================================================

This script processes author network data to:
1. Split data by time (pre-2016 = train, 2016+ = test)
2. Extract N-hop neighbors for each author
3. Compute common neighbor counts for candidate ranking

Usage:
    python graph_data_prep.py --cutoff-year 2016 --max-hops 3 --output-dir ./split_data
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphDataPreparation:
    """Prepare author network data for graph-based candidate generation."""
    
    def __init__(self, data_dir: str, cutoff_year: int = 2016, max_hops: int = 3):
        self.data_dir = Path(data_dir)
        self.cutoff_year = cutoff_year
        self.max_hops = max_hops
        
        # Data containers
        self.authors_df = None
        self.works_df = None
        self.coauthor_edges_df = None
        
        # Processed data
        self.work_id_to_year = {}
        self.train_coauthor_graph = defaultdict(set)  # {author_id: set(neighbors)}
        self.test_coauthor_graph = defaultdict(set)
        
    def load_data(self):
        """Load all parquet files."""
        logger.info("Loading data files...")
        
        self.authors_df = pd.read_parquet(self.data_dir / 'authors.parquet')
        self.works_df = pd.read_parquet(self.data_dir / 'collected_works_filtered.parquet')
        self.coauthor_edges_df = pd.read_parquet(self.data_dir / 'coauthor_edges.parquet')
        
        logger.info(f"Loaded {len(self.authors_df)} authors")
        logger.info(f"Loaded {len(self.works_df)} works")
        logger.info(f"Loaded {len(self.coauthor_edges_df)} coauthor edges")
        
    def extract_work_years(self):
        """Extract publication years from works."""
        logger.info("Extracting publication years...")
        
        # Convert publication_date to year
        self.works_df['year'] = pd.to_datetime(
            self.works_df['publication_date'], 
            errors='coerce'
        ).dt.year
        
        # Create work_id -> year mapping
        self.work_id_to_year = dict(zip(
            self.works_df['work_id'], 
            self.works_df['year']
        ))
        
        year_distribution = self.works_df['year'].value_counts().sort_index()
        logger.info(f"Year range: {self.works_df['year'].min()} - {self.works_df['year'].max()}")
        logger.info(f"Works before {self.cutoff_year}: {(self.works_df['year'] < self.cutoff_year).sum()}")
        logger.info(f"Works in/after {self.cutoff_year}: {(self.works_df['year'] >= self.cutoff_year).sum()}")
        
    def get_edge_year(self, work_ids_str: str) -> int:
        """Get the earliest year from a list of work IDs (for coauthor edges)."""
        try:
            # Parse work_ids from string representation of list
            work_ids = eval(work_ids_str) if isinstance(work_ids_str, str) else work_ids_str
            years = [self.work_id_to_year.get(wid) for wid in work_ids]
            return int(min(years)) if years else None
        except:
            return None

    
    def split_edges_by_time(self):
        """Split edges into train/test based on temporal cutoff."""
        logger.info(f"Splitting edges by year {self.cutoff_year}...")
        
        # Process coauthor edges
        logger.info("Processing coauthor edges...")
        train_coauthor = []
        test_coauthor = []
        
        for idx, row in self.coauthor_edges_df.iterrows():
            year = self.get_edge_year(row['work_ids']) #first year collaborated
            
            if year is None:
                continue
                
            edge_data = {
                'author_id_1': row['author_id_1'],
                'author_id_2': row['author_id_2'],
                'weight': row['weight'],
                'year': year
            }
            
            if year < self.cutoff_year:
                train_coauthor.append(edge_data)
                # Add to train graph (undirected)
                self.train_coauthor_graph[row['author_id_1']].add(row['author_id_2'])
                self.train_coauthor_graph[row['author_id_2']].add(row['author_id_1'])
            else:
                test_coauthor.append(edge_data)
                # Add to test graph (undirected)
                self.test_coauthor_graph[row['author_id_1']].add(row['author_id_2'])
                self.test_coauthor_graph[row['author_id_2']].add(row['author_id_1'])
        
        self.train_coauthor_df = pd.DataFrame(train_coauthor)
        self.test_coauthor_df = pd.DataFrame(test_coauthor)
        
        logger.info(f"Train coauthor edges: {len(self.train_coauthor_df)}")
        logger.info(f"Test coauthor edges: {len(self.test_coauthor_df)}")
        
    
    
    def compute_n_hop_neighbors(self, graph: Dict[str, Set[str]], source: str, n: int) -> Dict[int, Set[str]]:
        """
        Compute N-hop neighbors for a given source node using BFS.
        
        Returns:
            Dict mapping hop distance to set of neighbors at that distance
            {1: {neighbors at 1-hop}, 2: {neighbors at 2-hop}, ...}
        """
        if source not in graph:
            return {}
        
        visited = {source}
        neighbors_by_hop = defaultdict(set)
        queue = deque([(source, 0)])
        
        while queue:
            node, hop = queue.popleft()
            
            if hop >= n:
                continue
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbors_by_hop[hop + 1].add(neighbor)
                    queue.append((neighbor, hop + 1))
        
        return dict(neighbors_by_hop)
    
    def compute_all_n_hop_neighbors(self, graph: Dict[str, Set[str]], max_n: int = 3) -> Dict[str, Dict[int, List[str]]]:
        """Compute N-hop neighbors for all nodes in the graph."""
        logger.info(f"Computing {max_n}-hop neighbors for all authors...")
        
        n_hop_neighbors = {}
        total_authors = len(graph)
        
        for i, author_id in enumerate(graph.keys()):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{total_authors} authors...")
            
            neighbors_by_hop = self.compute_n_hop_neighbors(graph, author_id, max_n)
            # Convert sets to lists for JSON serialization
            n_hop_neighbors[author_id] = {
                hop: list(neighbors) for hop, neighbors in neighbors_by_hop.items()
            }
        
        logger.info(f"Completed N-hop neighbor computation for {len(n_hop_neighbors)} authors")
        return n_hop_neighbors
    
    def compute_common_neighbors(self, graph: Dict[str, Set[str]], user: str, candidates: List[str]) -> Dict[str, int]:
        """
        Compute the number of common neighbors between a user and each candidate.
        
        Args:
            graph: The graph (adjacency list)
            user: The user/author ID
            candidates: List of candidate author IDs
            
        Returns:
            Dict mapping candidate ID to common neighbor count
        """
        user_neighbors = graph.get(user, set())
        
        common_counts = {}
        for candidate in candidates:
            candidate_neighbors = graph.get(candidate, set())
            common_neighbors = user_neighbors & candidate_neighbors
            common_counts[candidate] = len(common_neighbors)
        
        return common_counts
    
    def save_data(self, output_dir: Path):
        """Save all processed data to output directory."""
        logger.info(f"Saving data to {output_dir}...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        train_dir = output_dir / 'train'
        test_dir = output_dir / 'test'
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Save authors (same for both train and test)
        self.authors_df.to_parquet(output_dir / 'authors.parquet', index=False)
        logger.info(f"Saved authors to {output_dir / 'authors.parquet'}")
        
        # Save train edges
        self.train_coauthor_df.to_parquet(train_dir / 'coauthor_edges.parquet', index=False)
        logger.info(f"Saved train edges")
        
        # Save test edges
        self.test_coauthor_df.to_parquet(test_dir / 'coauthor_edges.parquet', index=False)
        logger.info(f"Saved test edges")
        
        # Compute and save N-hop neighbors for train graph
        train_n_hop = self.compute_all_n_hop_neighbors(self.train_coauthor_graph, self.max_hops)
        with open(train_dir / 'n_hop_neighbors.json', 'w') as f:
            json.dump(train_n_hop, f, indent=2)
        logger.info(f"Saved train N-hop neighbors")
        
        # Compute and save N-hop neighbors for combined graph (train + test)
        # This is useful for evaluation
        combined_graph = defaultdict(set)
        for author, neighbors in self.train_coauthor_graph.items():
            combined_graph[author].update(neighbors)
        for author, neighbors in self.test_coauthor_graph.items():
            combined_graph[author].update(neighbors)
        
        test_n_hop = self.compute_all_n_hop_neighbors(combined_graph, self.max_hops)
        with open(test_dir / 'n_hop_neighbors.json', 'w') as f:
            json.dump(test_n_hop, f, indent=2)
        logger.info(f"Saved test N-hop neighbors")
        
        # Save adjacency lists (more compact format)
        # Convert sets to lists for JSON serialization
        train_coauthor_adj = {k: list(v) for k, v in self.train_coauthor_graph.items()}
        test_coauthor_adj = {k: list(v) for k, v in self.test_coauthor_graph.items()}
        combined_adj = {k: list(v) for k, v in combined_graph.items()}
        
        with open(train_dir / 'coauthor_adjlist.json', 'w') as f:
            json.dump(train_coauthor_adj, f)
        with open(test_dir / 'coauthor_adjlist.json', 'w') as f:
            json.dump(combined_adj, f)
        logger.info(f"Saved adjacency lists")
        
        # Save metadata and statistics
        metadata = {
            'cutoff_year': self.cutoff_year,
            'max_hops': self.max_hops,
            'created_at': datetime.now().isoformat(),
            'statistics': {
                'total_authors': len(self.authors_df),
                'total_works': len(self.works_df),
                'train': {
                    'coauthor_edges': len(self.train_coauthor_df),
                    'unique_authors_in_coauthor_network': len(self.train_coauthor_graph),
                    'min_year': int(self.train_coauthor_df['year'].min()) if len(self.train_coauthor_df) > 0 else None,
                    'max_year': int(self.train_coauthor_df['year'].max()) if len(self.train_coauthor_df) > 0 else None,
                },
                'test': {
                    'coauthor_edges': len(self.test_coauthor_df),
                    'unique_authors_in_coauthor_network': len(self.test_coauthor_graph),
                    'min_year': int(self.test_coauthor_df['year'].min()) if len(self.test_coauthor_df) > 0 else None,
                    'max_year': int(self.test_coauthor_df['year'].max()) if len(self.test_coauthor_df) > 0 else None,
                }
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"\nTrain/Test Split (cutoff year: {self.cutoff_year}):")
        logger.info(f"  Train coauthor edges: {len(self.train_coauthor_df)}")
        logger.info(f"  Test coauthor edges: {len(self.test_coauthor_df)}")
        logger.info(f"\nN-hop neighbors computed up to {self.max_hops} hops")
        logger.info("="*60)
    
    def run(self, output_dir: str):
        """Run the complete data preparation pipeline."""
        self.load_data()
        self.extract_work_years()
        self.split_edges_by_time()
        self.save_data(Path(output_dir))


def get_candidates_with_ranking(
    graph: Dict[str, Set[str]], 
    user_id: str, 
    n_hop_neighbors: Dict[str, Dict[int, List[str]]],
    max_candidates: int = 300
) -> List[Tuple[str, int, int]]:
    """
    Get candidate authors for a given user with ranking based on common neighbors.
    
    Args:
        graph: The coauthor graph (adjacency list as dict)
        user_id: The user/author ID to generate candidates for
        n_hop_neighbors: Pre-computed N-hop neighbors
        max_candidates: Maximum number of candidates to return
        
    Returns:
        List of (candidate_id, hop_distance, common_neighbor_count) sorted by common neighbors
    """
    if user_id not in n_hop_neighbors:
        return []
    
    user_neighbors = graph.get(user_id, set())
    candidates_with_scores = []
    
    # Get candidates from 2-hop and 3-hop neighbors
    for hop in [2, 3]:
        if hop in n_hop_neighbors[user_id]:
            for candidate in n_hop_neighbors[user_id][hop]:
                # Compute common neighbors
                candidate_neighbors = graph.get(candidate, set())
                common_count = len(user_neighbors & candidate_neighbors)
                candidates_with_scores.append((candidate, hop, common_count))
    
    # Sort by common neighbor count (descending), then by hop distance (ascending)
    candidates_with_scores.sort(key=lambda x: (-x[2], x[1]))
    
    return candidates_with_scores[:max_candidates]


def main():
    parser = argparse.ArgumentParser(description='Prepare graph data for candidate generation')
    parser.add_argument('--data-dir', type=str, default='/home/vo43/PYMK/OpenAlex',
                        help='Directory containing the parquet files')
    parser.add_argument('--cutoff-year', type=int, default=2019,
                        help='Year to split train/test (default: 2019)')
    parser.add_argument('--max-hops', type=int, default=3,
                        help='Maximum hops for N-hop neighbor computation (default: 3)')
    parser.add_argument('--output-dir', type=str, default='./split_data',
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    prep = GraphDataPreparation(
        data_dir=args.data_dir,
        cutoff_year=args.cutoff_year,
        max_hops=args.max_hops
    )
    
    prep.run(args.output_dir)


if __name__ == '__main__':
    main()
