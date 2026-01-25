"""
Candidate Generation Example
=============================

This script demonstrates how to use the processed graph data for
author candidate generation and ranking using N-hop neighbors and
common neighbor counting.

Usage:
    python candidate_generation_example.py --user-id <AUTHOR_ID>
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set


class CandidateGenerator:
    """Generate and rank author candidates for recommendation."""
    
    def __init__(self, data_dir: str = './split_data'):
        self.data_dir = Path(data_dir)
        
        # Load data
        self.authors_df = pd.read_parquet(self.data_dir / 'authors.parquet')
        
        # Load train graph (for generating candidates)
        with open(self.data_dir / 'train' / 'coauthor_adjlist.json', 'r') as f:
            self.train_graph = json.load(f)
        
        # Load N-hop neighbors (pre-computed)
        with open(self.data_dir / 'train' / 'n_hop_neighbors.json', 'r') as f:
            self.train_n_hop = json.load(f)
        
        # Load test graph (for evaluation - contains all edges)
        with open(self.data_dir / 'test' / 'coauthor_adjlist.json', 'r') as f:
            self.test_graph = json.load(f)
        
        # Load test edges for evaluation
        self.test_edges_df = pd.read_parquet(self.data_dir / 'test' / 'coauthor_edges.parquet')
        
        print(f"Loaded {len(self.authors_df)} authors")
        print(f"Train graph has {len(self.train_graph)} nodes")
        print(f"Test graph has {len(self.test_graph)} nodes")
    
    def get_candidates_with_ranking(
        self, 
        user_id: str, 
        hops: List[int] = [2, 3],
        max_candidates: int = 100,
        exclude_existing: bool = True
    ) -> List[Tuple[str, int, int]]:
        """
        Get candidate authors for a given user with ranking.
        
        Args:
            user_id: The user/author ID to generate candidates for
            hops: Which hop distances to consider (default: [2, 3])
            max_candidates: Maximum number of candidates to return
            exclude_existing: Whether to exclude existing 1-hop neighbors
            
        Returns:
            List of (candidate_id, hop_distance, common_neighbor_count)
            sorted by common neighbors (descending) then hop distance (ascending)
        """
        if user_id not in self.train_n_hop:
            print(f"Warning: User {user_id} not found in training graph")
            return []
        
        # Get existing neighbors (to exclude them from candidates)
        existing_neighbors = set(self.train_graph.get(user_id, []))
        
        # Get user's 1-hop neighbors for common neighbor calculation
        user_neighbors = set(self.train_graph.get(user_id, []))
        
        candidates_with_scores = []
        
        # Collect candidates from specified hops
        for hop in hops:
            hop_str = str(hop)  # JSON keys are strings
            if hop_str in self.train_n_hop[user_id]:
                for candidate in self.train_n_hop[user_id][hop_str]:
                    # Skip if already a neighbor
                    if exclude_existing and candidate in existing_neighbors:
                        continue
                    
                    # Skip self
                    if candidate == user_id:
                        continue
                    
                    # Calculate common neighbors (Jaccard or raw count)
                    candidate_neighbors = set(self.train_graph.get(candidate, []))
                    common_neighbors = user_neighbors & candidate_neighbors # set intersection
                    common_count = len(common_neighbors)
                    
                    candidates_with_scores.append((candidate, hop, common_count))
        
        # Sort by common neighbor count (descending), then by hop distance (ascending)
        candidates_with_scores.sort(key=lambda x: (-x[2], x[1]))
        
        return candidates_with_scores[:max_candidates]
    
    def get_author_info(self, author_id: str) -> Dict:
        """Get author information."""
        author = self.authors_df[self.authors_df['author_id'] == author_id]
        if len(author) == 0:
            return None
        
        return {
            'author_id': author_id,
            'name': author.iloc[0]['author_name'],
            'institution': author.iloc[0]['institution'],
            'work_count': author.iloc[0]['work_count'],
            'dominant_topic': author.iloc[0]['dominant_topic']
        }
    
    def evaluate_candidates(self, user_id: str, candidates: List[Tuple[str, int, int]]) -> Dict:
        """
        Evaluate candidate quality by checking if they appear in test edges.
        
        This tells us if the candidates we recommend actually became collaborators
        in the test period.
        """
        # Get actual new collaborators from test set
        test_neighbors = set(self.test_graph.get(user_id, [])) - set(self.train_graph.get(user_id, [])) # note: subtracting an empty set gives an empty set. 
        # Check how many of our candidates are actual new collaborators
        candidate_ids = [c[0] for c in candidates]
        hits = [c for c in candidate_ids if c in test_neighbors]
        return {
            'total_candidates': len(candidates),
            'actual_new_collaborators': len(test_neighbors),
            'hits': len(hits),
            'hit_rate': len(hits) / len(candidates) if candidates else 0,
            'recall': len(hits) / len(test_neighbors) if test_neighbors else 0,
            'hit_candidates': hits[:10]  # Show top 10 hits
        }
    
    def print_candidates(self, user_id: str, candidates: List[Tuple[str, int, int]], top_k: int = 10):
        """Pretty print candidate recommendations."""
        print(f"\n{'='*80}")
        print(f"CANDIDATE RECOMMENDATIONS FOR USER: {user_id}")
        print(f"{'='*80}\n")
        
        # User info
        user_info = self.get_author_info(user_id)
        if user_info:
            print(f"User: {user_info['name']}")
            print(f"Institution: {user_info['institution']}")
            print(f"Dominant Topic: {user_info['dominant_topic']}")
            print(f"Work Count: {user_info['work_count']}")
        
        print(f"\nTop {top_k} Candidates (ranked by common neighbors):")
        print(f"{'-'*80}")
        print(f"{'Rank':<6} {'Hop':<6} {'Common':<10} {'Name':<30} {'Topic':<20}")
        print(f"{'-'*80}")
        
        for i, (candidate_id, hop, common_count) in enumerate(candidates[:top_k], 1):
            cand_info = self.get_author_info(candidate_id)
            if cand_info:
                name = cand_info['name'][:28]
                topic = str(cand_info['dominant_topic'])[:18] if cand_info['dominant_topic'] else 'N/A'
                print(f"{i:<6} {hop:<6} {common_count:<10} {name:<30} {topic:<20}")
        
        print(f"{'-'*80}\n")
        
        # Evaluation
        eval_results = self.evaluate_candidates(user_id, candidates)
        print("Evaluation (against test set):")
        print(f"  Total candidates recommended: {eval_results['total_candidates']}")
        print(f"  Actual new collaborators in test: {eval_results['actual_new_collaborators']}")
        print(f"  Hits (correct predictions): {eval_results['hits']}")
        print(f"  Hit rate: {eval_results['hit_rate']:.2%}")
        print(f"  Recall: {eval_results['recall']:.2%}")
        
        if eval_results['hit_candidates']:
            print(f"\n  Sample correct predictions:")
            for hit_id in eval_results['hit_candidates'][:5]:
                hit_info = self.get_author_info(hit_id)
                if hit_info:
                    print(f"    - {hit_info['name']} ({hit_info['dominant_topic']})")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate author candidates')
    parser.add_argument('--data-dir', type=str, default='./split_data',
                        help='Directory containing processed data')
    parser.add_argument('--user-id', type=str, required=False,
                        help='User author ID to generate candidates for')
    parser.add_argument('--hops', type=int, nargs='+', default=[2, 3],
                        help='Which hop distances to consider (default: 2 3)')
    parser.add_argument('--max-candidates', type=int, default=100,
                        help='Maximum number of candidates (default: 50)')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of top candidates to display (default: 15)')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with a random user')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CandidateGenerator(data_dir=args.data_dir)
    
    # Get user ID
    if args.demo or not args.user_id:
        # Pick a random user with some collaborations
        import random
        users_with_neighbors = [u for u in generator.train_graph.keys()  
                                if generator.get_author_info(u)['work_count'] >= 3 and len(generator.test_graph[u]) > len(generator.train_graph[u])]
        user_id = random.choice(users_with_neighbors) if users_with_neighbors else exit("No users with neighbors found in test set")
        #user_id = "https://openalex.org/A5014486069"
        #print(f"User has {len(generator.train_graph[user_id])} neighbors in train graph")
        print(f"Demo mode: Using random user {user_id}\n")
    else:
        user_id = args.user_id
    
    # Generate candidates
    candidates = generator.get_candidates_with_ranking(
        user_id=user_id,
        hops=args.hops,
        max_candidates=args.max_candidates
    )
    
    if not candidates:
        print(f"No candidates found for user {user_id}")
        return
    
    # Print results
    generator.print_candidates(user_id, candidates, top_k=args.top_k)
    
    # Show statistics
    print("\nCandidate Distribution by Hop:")
    hop_counts = {}
    for _, hop, _ in candidates:
        hop_counts[hop] = hop_counts.get(hop, 0) + 1
    for hop in sorted(hop_counts.keys()):
        print(f"  {hop}-hop: {hop_counts[hop]} candidates")
    
    print("\nCommon Neighbor Statistics:")
    common_counts = [c[2] for c in candidates]
    if common_counts:
        print(f"  Min: {min(common_counts)}")
        print(f"  Max: {max(common_counts)}")
        print(f"  Mean: {sum(common_counts) / len(common_counts):.2f}")
        print(f"  Median: {sorted(common_counts)[len(common_counts)//2]}")


if __name__ == '__main__':
    main()
