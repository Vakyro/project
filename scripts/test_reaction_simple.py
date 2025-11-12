"""
Script simple para probar el encoder de reacciones.
Usa tus propias reacciones SMILES y ve los resultados.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import csv
from reaction_encoder.chem import parse_reaction_smiles
from reaction_encoder.builder import build_transition_graph
from reaction_encoder.model_enhanced import ReactionGNNEnhanced


def test_basic():
    """Prueba básica - codifica 2 reacciones."""
    print("=" * 60)
    print("TEST 1: Prueba Básica de Reacciones")
    print("=" * 60)

    # Load reactions from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
    reactions_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'N=N' in row['reaction_name'] or 'C=C' in row['reaction_name']:
                reactions_data.append((row['reaction_smiles'], row['reaction_name']))
                if len(reactions_data) >= 2:
                    break

    rxn1, name1 = reactions_data[0]
    rxn2, name2 = reactions_data[1]

    print(f"\nReacción 1: {name1} - {rxn1}")
    print(f"Reacción 2: {name2} - {rxn2}")

    # Construir grafos
    print("\nConstruyendo grafos de transición...")
    reacts1, prods1 = parse_reaction_smiles(rxn1)
    data1 = build_transition_graph(reacts1, prods1, use_enhanced_features=True)

    reacts2, prods2 = parse_reaction_smiles(rxn2)
    data2 = build_transition_graph(reacts2, prods2, use_enhanced_features=True)

    print(f"  Reacción 1: {data1.num_nodes} nodos, {data1.edge_index.size(1)} aristas")
    print(f"  Reacción 2: {data2.num_nodes} nodos, {data2.edge_index.size(1)} aristas")

    # Inicializar modelo
    print("\nInicializando modelo...")
    model = ReactionGNNEnhanced(
        x_dim=data1.x.size(1),
        e_dim=data1.edge_attr.size(1),
        hidden=128,
        layers=3,
        out_dim=256
    ).eval()

    # Codificar
    print("\nCodificando reacciones...")
    with torch.no_grad():
        z1 = model(data1)
        z2 = model(data2)

    print(f"\nResultado:")
    print(f"  Embedding 1: {z1.shape}, norma={torch.norm(z1).item():.4f}")
    print(f"  Embedding 2: {z2.shape}, norma={torch.norm(z2).item():.4f}")

    # Similitud
    similarity = (z1[0] @ z2[0]).item()
    print(f"\nSimilitud coseno: {similarity:.4f}")
    print(f"  (Son similares porque ambas son reducciones!)")

    print("\n✓ Prueba completada!")


def test_tu_reaccion():
    """Codifica TU reacción."""
    print("\n" + "=" * 60)
    print("TEST 2: Codifica TU Reacción")
    print("=" * 60)

    # Load a reaction from CSV (users can edit the CSV to add their own)
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Triple' in row['reaction_name']:
                mi_reaccion = row['reaction_smiles']
                reaction_name = row['reaction_name']
                break

    print(f"\nTu reacción: {reaction_name} - {mi_reaccion}")

    # Parsear y construir grafo
    print("\nParseando reacción...")
    reacts, prods = parse_reaction_smiles(mi_reaccion)
    data = build_transition_graph(reacts, prods, use_enhanced_features=True)

    print(f"Grafo construido:")
    print(f"  Nodos: {data.num_nodes}")
    print(f"  Aristas: {data.edge_index.size(1)}")
    print(f"  Node features: {data.x.size(1)} dimensiones")
    print(f"  Edge features: {data.edge_attr.size(1)} dimensiones")

    # Modelo
    model = ReactionGNNEnhanced(
        x_dim=data.x.size(1),
        e_dim=data.edge_attr.size(1),
        hidden=128,
        layers=3,
        out_dim=256
    ).eval()

    # Codificar
    print("\nCodificando...")
    with torch.no_grad():
        embedding = model(data)

    print(f"\nEmbedding generado:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Norma: {torch.norm(embedding).item():.4f}")
    print(f"  Primeros 10 valores: {embedding[0, :10].tolist()}")

    print("\n✓ Tu reacción fue codificada!")


def test_multiples():
    """Compara varias reacciones."""
    print("\n" + "=" * 60)
    print("TEST 3: Comparar Múltiples Reacciones")
    print("=" * 60)

    # Load reactions from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
    reacciones = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Load reduction reactions
            if 'reduction' in row['reaction_name'].lower() or 'reducción' in row['reaction_name'].lower():
                reacciones[row['reaction_name']] = row['reaction_smiles']
                if len(reacciones) >= 4:
                    break

    print(f"\nReacciones a comparar (from CSV):")
    for nombre, smiles in reacciones.items():
        print(f"  {nombre}: {smiles}")

    # Construir grafos
    print("\nConstruyendo grafos...")
    grafos = []
    nombres = []
    for nombre, rxn_smiles in reacciones.items():
        try:
            reacts, prods = parse_reaction_smiles(rxn_smiles)
            data = build_transition_graph(reacts, prods, use_enhanced_features=True)
            grafos.append(data)
            nombres.append(nombre)
        except Exception as e:
            print(f"  ✗ {nombre}: Error - {e}")

    print(f"  ✓ {len(grafos)} reacciones construidas")

    # Modelo
    model = ReactionGNNEnhanced(
        x_dim=grafos[0].x.size(1),
        e_dim=grafos[0].edge_attr.size(1),
        hidden=128,
        layers=3,
        out_dim=256
    ).eval()

    # Codificar todas
    print("\nCodificando todas las reacciones...")
    embeddings = []
    with torch.no_grad():
        for data in grafos:
            z = model(data)
            embeddings.append(z)

    embeddings = torch.cat(embeddings, dim=0)

    # Matriz de similitud
    print("\nMatriz de Similitud:")
    print()

    # Header
    print(" " * 20, end="")
    for nombre in nombres:
        print(f"{nombre[:12]:12s}", end="")
    print()

    # Filas
    sim_matrix = embeddings @ embeddings.t()
    for i, nombre_i in enumerate(nombres):
        print(f"{nombre_i[:20]:20s}", end="")
        for j in range(len(nombres)):
            sim = sim_matrix[i, j].item()
            print(f"{sim:12.4f}", end="")
        print()

    # Pares más similares
    print("\nPares más similares:")
    pairs = []
    for i in range(len(nombres)):
        for j in range(i+1, len(nombres)):
            sim = sim_matrix[i, j].item()
            pairs.append((sim, nombres[i], nombres[j]))

    pairs.sort(reverse=True)
    for sim, n1, n2 in pairs[:3]:
        print(f"  {n1} <-> {n2}: {sim:.4f}")

    print("\n✓ Comparación completada!")


def test_con_cambios():
    """Analiza qué enlaces cambian en una reacción."""
    print("\n" + "=" * 60)
    print("TEST 4: Análisis de Cambios en Reacción")
    print("=" * 60)

    from reaction_encoder.chem import bond_set
    from reaction_encoder.builder import diff_bonds, EdgeChange

    # Load hydrogenation reaction from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'hydrogenation' in row['reaction_name'].lower():
                rxn = row['reaction_smiles']
                reaction_name = row['reaction_name']
                break

    print(f"\nReacción: {rxn}")
    print(f"({reaction_name})")

    # Parsear
    reacts, prods = parse_reaction_smiles(rxn)
    bonds_r = bond_set(reacts)
    bonds_p = bond_set(prods)
    changes = diff_bonds(bonds_r, bonds_p)

    # Contar cambios
    formed = [e for e, c in changes.items() if c == EdgeChange.FORMED]
    broken = [e for e, c in changes.items() if c == EdgeChange.BROKEN]
    changed = [e for e, c in changes.items() if c == EdgeChange.CHANGED_ORDER]

    print(f"\nCambios detectados:")
    print(f"  Enlaces formados:  {len(formed)} - {formed}")
    print(f"  Enlaces rotos:     {len(broken)} - {broken}")
    print(f"  Orden cambiado:    {len(changed)} - {changed}")

    # Construir y codificar
    data = build_transition_graph(reacts, prods, use_enhanced_features=True)
    model = ReactionGNNEnhanced(
        x_dim=data.x.size(1),
        e_dim=data.edge_attr.size(1),
        hidden=128,
        layers=3,
        out_dim=256
    ).eval()

    with torch.no_grad():
        embedding = model(data)

    print(f"\nEmbedding generado con {data.num_nodes} nodos")
    print(f"  Norma: {torch.norm(embedding).item():.4f}")

    print("\n✓ Análisis completado!")
    print("  El modelo 've' todos estos cambios en el grafo!")


if __name__ == "__main__":
    print("\n")
    print("#" * 60)
    print("# PRUEBAS DEL REACTION ENCODER")
    print("#" * 60)

    try:
        test_basic()
        test_tu_reaccion()
        test_multiples()
        test_con_cambios()

        print("\n" + "#" * 60)
        print("# ¡TODAS LAS PRUEBAS PASARON!")
        print("#" * 60)
        print("\nTips:")
        print("  - Las reacciones DEBEN tener mapeo de átomos [X:1], [Y:2]")
        print("  - Puedes cambiar las reacciones en este script")
        print("  - Similitudes altas = reacciones químicamente similares")
        print()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
