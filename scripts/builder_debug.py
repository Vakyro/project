"""
Debug script to verify reaction changes (bonds formed/broken/changed).
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reaction_encoder.chem import parse_reaction_smiles, bond_set, mapnums_index
from reaction_encoder.builder import diff_bonds, EdgeChange


def debug_reaction_changes(rxn, name=""):
    """Debug a single reaction to see what changes."""
    print("\n" + "=" * 70)
    if name:
        print(f"Reaction: {name}")
    print(f"SMILES: {rxn}")
    print("=" * 70)

    try:
        reacts, prods = parse_reaction_smiles(rxn)
        idx_r = mapnums_index(reacts)
        idx_p = mapnums_index(prods)
        bonds_r = bond_set(reacts)
        bonds_p = bond_set(prods)
        changes = diff_bonds(bonds_r, bonds_p)

        formed = [e for e, c in changes.items() if c == EdgeChange.FORMED]
        broken = [e for e, c in changes.items() if c == EdgeChange.BROKEN]
        changed = [e for e, c in changes.items() if c == EdgeChange.CHANGED_ORDER]
        unchanged = [e for e, c in changes.items() if c == EdgeChange.UNCHANGED]

        print(f"\nAtoms in reactants: {len(idx_r)} mapped atoms")
        print(f"Atoms in products:  {len(idx_p)} mapped atoms")
        print(f"\nBonds in reactants: {len(bonds_r)}")
        print(f"Bonds in products:  {len(bonds_p)}")

        print(f"\n--- Bond Changes ---")
        print(f"FORMED:         {len(formed):3d} bonds {formed[:5]}")
        print(f"BROKEN:         {len(broken):3d} bonds {broken[:5]}")
        print(f"CHANGED_ORDER:  {len(changed):3d} bonds {changed[:5]}")
        print(f"UNCHANGED:      {len(unchanged):3d} bonds")

        # Show bond details for changes
        if formed:
            print("\n  Formed bonds (detail):")
            for u, v in formed[:5]:
                bond_info = bonds_p.get((u, v), bonds_p.get((v, u), {}))
                print(f"    {u:3d}--{v:3d}  order={bond_info.get('order', '?')}")

        if broken:
            print("\n  Broken bonds (detail):")
            for u, v in broken[:5]:
                bond_info = bonds_r.get((u, v), bonds_r.get((v, u), {}))
                print(f"    {u:3d}--{v:3d}  order={bond_info.get('order', '?')}")

        if changed:
            print("\n  Changed bonds (detail):")
            for u, v in changed[:5]:
                bond_r = bonds_r.get((u, v), bonds_r.get((v, u), {}))
                bond_p = bonds_p.get((u, v), bonds_p.get((v, u), {}))
                print(f"    {u:3d}--{v:3d}  {bond_r.get('order', '?')} -> {bond_p.get('order', '?')}")

        return {
            'formed': len(formed),
            'broken': len(broken),
            'changed': len(changed),
            'unchanged': len(unchanged)
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def debug_csv_reactions(csv_path):
    """Debug all reactions in a CSV file."""
    import csv

    print("\n" + "#" * 70)
    print("# DEBUG: Reaction Bond Changes")
    print("#" * 70)

    reactions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reactions.append({
                'smiles': row['reaction_smiles'],
                'name': row['reaction_name']
            })

    stats = []
    for rxn in reactions:
        result = debug_reaction_changes(rxn['smiles'], rxn['name'])
        if result:
            stats.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total reactions: {len(stats)}")
    if stats:
        avg_formed = sum(s['formed'] for s in stats) / len(stats)
        avg_broken = sum(s['broken'] for s in stats) / len(stats)
        avg_changed = sum(s['changed'] for s in stats) / len(stats)
        print(f"Avg FORMED:   {avg_formed:.1f} bonds/reaction")
        print(f"Avg BROKEN:   {avg_broken:.1f} bonds/reaction")
        print(f"Avg CHANGED:  {avg_changed:.1f} bonds/reaction")

        # Warning if too few changes
        if avg_formed + avg_broken + avg_changed < 1.0:
            print("\n⚠️  WARNING: Very few bond changes detected!")
            print("    Consider using reactions with more obvious transformations.")
    print("=" * 70)


if __name__ == "__main__":
    # Test with CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions.csv')
    if os.path.exists(csv_path):
        debug_csv_reactions(csv_path)
    else:
        print(f"CSV not found at {csv_path}")

        # Test with reactions from reactions_extended.csv as fallback
        csv_path_extended = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
        if os.path.exists(csv_path_extended):
            print(f"\nTrying extended reactions CSV: {csv_path_extended}")
            import csv as csv_module
            with open(csv_path_extended, 'r', encoding='utf-8') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    if any(x in row['reaction_name'] for x in ['N=N', 'C≡C', 'Carbonyl']):
                        debug_reaction_changes(row['reaction_smiles'], row['reaction_name'])
        else:
            print("No CSV files found")
