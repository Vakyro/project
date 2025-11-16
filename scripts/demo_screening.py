#!/usr/bin/env python3
"""
CLIPZyme Screening Demo

Demonstrates all features of the screening system with example data.

Run all demos:
    python scripts/demo_screening.py --demo all

Run specific demo:
    python scripts/demo_screening.py --demo interactive
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("CLIPZyme Screening System Demo")
print("=" * 80)


def demo_1_screening_set():
    """Demo 1: Working with ScreeningSet"""
    print("\n" + "=" * 80)
    print("Demo 1: ScreeningSet - Managing Pre-Embedded Proteins")
    print("=" * 80)

    from screening.screening_set import ScreeningSet

    # Create a screening set
    screening_set = ScreeningSet(embedding_dim=512, device="cpu")

    # Add some example embeddings
    print("\n1. Adding protein embeddings...")
    for i in range(100):
        protein_id = f"P{i:05d}"
        # Random normalized embedding
        embedding = torch.randn(512)
        embedding = embedding / embedding.norm()
        screening_set.add_embedding(protein_id, embedding)

    print(f"✓ Added {len(screening_set)} proteins")

    # Get embeddings as tensor
    print("\n2. Getting all embeddings...")
    embeddings, protein_ids = screening_set.get_embeddings_tensor()
    print(f"✓ Embeddings tensor shape: {embeddings.shape}")
    print(f"✓ First 5 protein IDs: {protein_ids[:5]}")

    # Compute similarity with a query
    print("\n3. Computing similarity...")
    query = torch.randn(512)
    query = query / query.norm()

    scores, indices, top_proteins = screening_set.compute_similarity(
        query,
        top_k=10,
        return_scores=True
    )

    print(f"✓ Top 10 matches:")
    for i, (pid, score) in enumerate(zip(top_proteins, scores), 1):
        print(f"   {i}. {pid}: {score:.4f}")

    # Save and load
    print("\n4. Saving screening set...")
    temp_file = "temp_screening_set.p"
    screening_set.save_to_pickle(temp_file)
    print(f"✓ Saved to {temp_file}")

    print("\n5. Loading screening set...")
    loaded_set = ScreeningSet(device="cpu")
    loaded_set.load_from_pickle(temp_file)
    print(f"✓ Loaded {len(loaded_set)} proteins")

    # Cleanup
    Path(temp_file).unlink()
    print(f"✓ Cleaned up temporary file")


def demo_2_protein_database():
    """Demo 2: Protein Database"""
    print("\n" + "=" * 80)
    print("Demo 2: ProteinDatabase - Managing Protein Sequences")
    print("=" * 80)

    from screening.screening_set import ProteinDatabase
    import pandas as pd

    # Create sample data
    print("\n1. Creating sample protein data...")
    protein_data = pd.DataFrame({
        'protein_id': [f'P{i:05d}' for i in range(50)],
        'sequence': ['MKVLWAALLVTFLAG' * (i % 5 + 1) for i in range(50)],
        'ec_number': [f'{i%6}.1.1.{i%10}' for i in range(50)],
        'organism': ['E. coli' if i % 2 == 0 else 'Human' for i in range(50)]
    })

    temp_csv = "temp_proteins.csv"
    protein_data.to_csv(temp_csv, index=False)
    print(f"✓ Created {len(protein_data)} sample proteins")

    # Load into database
    print("\n2. Loading protein database...")
    protein_db = ProteinDatabase()
    protein_db.load_from_csv(
        csv_path=temp_csv,
        id_column="protein_id",
        sequence_column="sequence",
        metadata_columns=["ec_number", "organism"]
    )
    print(f"✓ Loaded {len(protein_db)} proteins")

    # Query proteins
    print("\n3. Querying proteins...")
    protein_id = "P00010"
    sequence = protein_db.get_sequence(protein_id)
    print(f"✓ {protein_id} sequence: {sequence[:50]}...")

    entry = protein_db[protein_id]
    print(f"✓ EC number: {entry.metadata['ec_number']}")
    print(f"✓ Organism: {entry.metadata['organism']}")

    # Save to pickle
    print("\n4. Saving to pickle...")
    temp_pickle = "temp_proteins.p"
    protein_db.save_to_pickle(temp_pickle)
    print(f"✓ Saved to {temp_pickle}")

    # Cleanup
    Path(temp_csv).unlink()
    Path(temp_pickle).unlink()
    print(f"✓ Cleaned up temporary files")


def demo_3_ranking_metrics():
    """Demo 3: Ranking and Evaluation Metrics"""
    print("\n" + "=" * 80)
    print("Demo 3: Ranking Metrics - BEDROC, Top-K, Enrichment Factor")
    print("=" * 80)

    from screening.ranking import (
        compute_bedroc,
        compute_topk_accuracy,
        compute_enrichment_factor,
        evaluate_screening
    )

    # Simulate screening results
    print("\n1. Simulating screening results...")
    ranked_proteins = [f"P{i:05d}" for i in range(100)]
    scores = np.linspace(1.0, 0.0, 100)
    true_positives = ["P00000", "P00005", "P00010", "P00050"]  # 4 true hits
    labels = [1 if pid in true_positives else 0 for pid in ranked_proteins]

    print(f"✓ Ranked list: {len(ranked_proteins)} proteins")
    print(f"✓ True positives: {true_positives}")

    # BEDROC
    print("\n2. Computing BEDROC...")
    for alpha in [20.0, 50.0, 85.0]:
        bedroc = compute_bedroc(scores, labels, alpha=alpha)
        print(f"✓ BEDROC_{int(alpha)}: {bedroc:.4f}")

    # Top-K Accuracy
    print("\n3. Computing Top-K Accuracy...")
    for k in [1, 5, 10, 50]:
        acc = compute_topk_accuracy(ranked_proteins, true_positives, k)
        print(f"✓ Top{k} Accuracy: {acc:.4f}")

    # Enrichment Factor
    print("\n4. Computing Enrichment Factor...")
    for frac in [0.01, 0.05, 0.10]:
        ef = compute_enrichment_factor(ranked_proteins, true_positives, frac)
        print(f"✓ EF_{int(frac*100)}%: {ef:.2f}")

    # Comprehensive evaluation
    print("\n5. Comprehensive evaluation...")
    metrics = evaluate_screening(
        ranked_protein_ids=ranked_proteins,
        scores=torch.tensor(scores),
        true_positives=true_positives
    )

    print(f"✓ All metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


def demo_4_interactive_screener():
    """Demo 4: Interactive Screening"""
    print("\n" + "=" * 80)
    print("Demo 4: InteractiveScreener - Single Reaction Screening")
    print("=" * 80)

    from screening.screening_set import ScreeningSet
    from screening.interactive_mode import InteractiveScreener, InteractiveScreeningConfig
    from reaction_encoder.builder import build_transition_graph
    from reaction_encoder.chem import parse_reaction_smiles
    import torch.nn as nn

    print("\n1. Creating mock model and screening set...")

    # Mock reaction encoder
    class MockReactionEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 512)

        def encode(self, reactions, device="cpu"):
            # Return random embeddings
            batch_size = len(reactions)
            embeddings = torch.randn(batch_size, 512)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            return embeddings

    model = MockReactionEncoder()

    # Create screening set
    screening_set = ScreeningSet(embedding_dim=512, device="cpu")
    for i in range(200):
        protein_id = f"P{i:05d}"
        embedding = torch.randn(512)
        embedding = embedding / embedding.norm()
        screening_set.add_embedding(protein_id, embedding)

    print(f"✓ Created screening set with {len(screening_set)} proteins")

    # Create screener
    print("\n2. Creating interactive screener...")
    config = InteractiveScreeningConfig(top_k=10, device="cpu")
    screener = InteractiveScreener(
        model=(None, model),
        screening_set=screening_set,
        config=config
    )
    print(f"✓ {screener}")

    # Screen a reaction
    print("\n3. Screening a reaction...")
    reaction_smiles = "[C:1]=[O:2]>>[C:1]-[O:2]"
    result = screener.screen_reaction(reaction_smiles, top_k=10)

    print(f"✓ Reaction: {reaction_smiles}")
    print(f"✓ Top 10 matches:")
    for i, (pid, score) in enumerate(zip(result.ranked_protein_ids, result.scores), 1):
        print(f"   {i}. {pid}: {score:.4f}")

    # Compare reactions
    print("\n4. Comparing two reactions...")
    rxn1 = "[C:1]=[O:2]>>[C:1]-[O:2]"
    rxn2 = "[N:1]=[N:2]>>[N:1]-[N:2]"

    comparison = screener.compare_reactions(rxn1, rxn2, top_k=20)
    print(f"✓ Reaction similarity: {comparison['reaction_similarity']:.4f}")
    print(f"✓ Overlap: {comparison['overlap_count']} proteins")
    print(f"✓ Jaccard index: {comparison['jaccard_index']:.4f}")


def demo_5_embedding_cache():
    """Demo 5: Embedding Cache"""
    print("\n" + "=" * 80)
    print("Demo 5: EmbeddingCache - Caching for Performance")
    print("=" * 80)

    from screening.cache import EmbeddingCache, LRUCache
    import time

    # Create cache
    print("\n1. Creating embedding cache...")
    cache = EmbeddingCache(
        memory_cache_size=50,
        use_disk=False  # Memory only for demo
    )
    print(f"✓ {cache}")

    # Add embeddings
    print("\n2. Caching embeddings...")
    for i in range(100):
        protein_id = f"P{i:05d}"
        embedding = torch.randn(512)
        cache.put(protein_id, embedding)

    print(f"✓ Cached 100 embeddings (LRU keeps 50)")

    # Test cache hits/misses
    print("\n3. Testing cache performance...")
    start = time.time()

    hits = 0
    misses = 0

    # Query recent proteins (should be cached)
    for i in range(50, 100):
        protein_id = f"P{i:05d}"
        result = cache.get(protein_id)
        if result is not None:
            hits += 1
        else:
            misses += 1

    # Query old proteins (evicted from LRU)
    for i in range(0, 50):
        protein_id = f"P{i:05d}"
        result = cache.get(protein_id)
        if result is not None:
            hits += 1
        else:
            misses += 1

    elapsed = time.time() - start

    print(f"✓ Hits: {hits}")
    print(f"✓ Misses: {misses}")
    print(f"✓ Time: {elapsed*1000:.2f}ms")

    # Get statistics
    print("\n4. Cache statistics...")
    stats = cache.get_stats()
    if 'memory' in stats:
        print(f"✓ Memory cache:")
        print(f"   - Hits: {stats['memory']['hits']}")
        print(f"   - Misses: {stats['memory']['misses']}")
        print(f"   - Hit rate: {stats['memory']['hit_rate']:.2%}")
        print(f"   - Size: {stats['memory']['cache_size']}/{stats['memory']['max_size']}")


def demo_6_full_workflow():
    """Demo 6: Complete Screening Workflow"""
    print("\n" + "=" * 80)
    print("Demo 6: Full Workflow - End-to-End Screening")
    print("=" * 80)

    from screening.screening_set import ScreeningSet, ProteinDatabase, build_screening_set_from_model
    from screening.interactive_mode import InteractiveScreener
    from screening.ranking import evaluate_screening, batch_evaluate_screening
    import pandas as pd
    import torch.nn as nn

    # Step 1: Create protein database
    print("\n1. Creating protein database...")
    proteins = pd.DataFrame({
        'protein_id': [f'ENZ_{i:03d}' for i in range(100)],
        'sequence': ['MKVLWAALLVTFLAG' * 5 for _ in range(100)],
        'ec_number': [f'{i%6}.{i%4}.{i%3}.{i%10}' for i in range(100)]
    })

    temp_csv = "temp_workflow_proteins.csv"
    proteins.to_csv(temp_csv, index=False)

    protein_db = ProteinDatabase()
    protein_db.load_from_csv(temp_csv, id_column='protein_id', sequence_column='sequence')
    print(f"✓ Created database with {len(protein_db)} proteins")

    # Step 2: Build screening set (mock)
    print("\n2. Building screening set...")
    screening_set = ScreeningSet(embedding_dim=512, device="cpu")
    for pid in proteins['protein_id']:
        embedding = torch.randn(512)
        embedding = embedding / embedding.norm()
        screening_set.add_embedding(pid, embedding)
    print(f"✓ Built screening set with {len(screening_set)} proteins")

    # Step 3: Create reactions to screen
    print("\n3. Creating test reactions...")
    reactions = pd.DataFrame({
        'reaction_id': [f'RXN_{i:03d}' for i in range(10)],
        'reaction_smiles': [f'[C:{i}]=[O:1]>>[C:{i}]-[O:1]' for i in range(1, 11)],
        'known_enzymes': ['ENZ_000,ENZ_010' if i % 3 == 0 else 'ENZ_050' for i in range(10)]
    })

    temp_rxn_csv = "temp_workflow_reactions.csv"
    reactions.to_csv(temp_rxn_csv, index=False)
    print(f"✓ Created {len(reactions)} test reactions")

    # Step 4: Screen reactions
    print("\n4. Screening reactions...")

    class MockReactionEncoder(nn.Module):
        def encode(self, reactions, device="cpu"):
            batch_size = len(reactions)
            embeddings = torch.randn(batch_size, 512)
            return embeddings / embeddings.norm(dim=1, keepdim=True)

    model = MockReactionEncoder()
    screener = InteractiveScreener(model=(None, model), screening_set=screening_set)

    results = []
    for _, row in reactions.iterrows():
        true_pos = row['known_enzymes'].split(',')
        result = screener.screen_reaction(
            row['reaction_smiles'],
            reaction_id=row['reaction_id'],
            true_positives=true_pos,
            top_k=20
        )
        results.append(result)

    print(f"✓ Screened {len(results)} reactions")

    # Step 5: Evaluate results
    print("\n5. Evaluating results...")
    all_metrics = []
    for result in results:
        metrics = evaluate_screening(
            ranked_protein_ids=result.ranked_protein_ids,
            scores=result.scores,
            true_positives=result.true_positives
        )
        all_metrics.append(metrics)
        print(f"✓ {result.reaction_id}: BEDROC_20={metrics['BEDROC_20']:.3f}")

    # Aggregate metrics
    print("\n6. Aggregate metrics...")
    aggregate = batch_evaluate_screening(results)
    print(f"✓ Average BEDROC_20: {aggregate['BEDROC_20']:.4f}")
    print(f"✓ Average Top10 Acc: {aggregate['Top10_Accuracy']:.4f}")

    # Cleanup
    Path(temp_csv).unlink()
    Path(temp_rxn_csv).unlink()
    print(f"✓ Cleaned up temporary files")


def main():
    parser = argparse.ArgumentParser(description="CLIPZyme Screening Demo")
    parser.add_argument(
        '--demo',
        type=str,
        default='all',
        choices=['all', '1', '2', '3', '4', '5', '6',
                 'screening_set', 'database', 'metrics',
                 'interactive', 'cache', 'workflow'],
        help='Which demo to run'
    )

    args = parser.parse_args()

    demos = {
        '1': ('screening_set', demo_1_screening_set),
        '2': ('database', demo_2_protein_database),
        '3': ('metrics', demo_3_ranking_metrics),
        '4': ('interactive', demo_4_interactive_screener),
        '5': ('cache', demo_5_embedding_cache),
        '6': ('workflow', demo_6_full_workflow),
    }

    if args.demo == 'all':
        for num, (name, func) in demos.items():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Demo {num} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Find demo by number or name
        demo_func = None
        if args.demo in demos:
            demo_func = demos[args.demo][1]
        else:
            for num, (name, func) in demos.items():
                if name == args.demo:
                    demo_func = func
                    break

        if demo_func:
            demo_func()
        else:
            print(f"Unknown demo: {args.demo}")

    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
