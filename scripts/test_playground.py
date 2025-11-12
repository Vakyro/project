"""
PLAYGROUND: Prueba tus propias proteínas y reacciones.

Ahora los datos se cargan desde CSVs. Puedes editarlos en:
  - data/proteins.csv (para proteínas)
  - data/reactions_extended.csv (para reacciones)
O puedes descomentar las secciones abajo para usar datos inline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import csv
from protein_encoder import ProteinEncoderESM2
from reaction_encoder.chem import parse_reaction_smiles
from reaction_encoder.builder import build_transition_graph
from reaction_encoder.model_enhanced import ReactionGNNEnhanced


# ============================================================
# OPCIÓN 1: Cargar desde CSV (recomendado)
# ============================================================

def load_data_from_csv():
    """Load proteins and reactions from CSV files."""
    # Load proteins
    proteins_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'proteins.csv')
    MIS_PROTEINAS = {}
    with open(proteins_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['length']) <= 100:  # Load shorter proteins for demo
                MIS_PROTEINAS[row['name']] = row['sequence']
                if len(MIS_PROTEINAS) >= 3:
                    break

    # Load reactions
    reactions_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
    MIS_REACCIONES = {}
    with open(reactions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'reduction' in row['reaction_name'].lower():
                MIS_REACCIONES[row['reaction_name']] = row['reaction_smiles']
                if len(MIS_REACCIONES) >= 3:
                    break

    return MIS_PROTEINAS, MIS_REACCIONES

# ============================================================
# OPCIÓN 2: Datos inline (descomenta para usar)
# ============================================================

# MIS_PROTEINAS = {
#     "Mi enzima 1": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK",
#     "Mi enzima 2": "MAHHHHHHSLENPLKQFGPVVVNQQWKK",
#     "Mi enzima 3": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRR",
# }

# MIS_REACCIONES = {
#     "Reacción A": "[N:1]=[N:2]>>[N:1][N:2]",
#     "Reacción B": "[C:1]=[C:2]>>[C:1][C:2]",
#     "Reacción C": "[C:1]#[C:2]>>[C:1]=[C:2]",
# }

# ============================================================


def playground():
    """Ejecuta el matching de proteínas vs reacciones."""

    print("=" * 70)
    print("PLAYGROUND: Enzyme-Reaction Matching")
    print("=" * 70)

    device = "cpu"  # Cambia a "cuda" si tienes GPU

    # ========== Load data ==========
    # Check if inline data is defined, otherwise load from CSV
    try:
        MIS_PROTEINAS
        MIS_REACCIONES
        print("\n[Using inline data from script]")
    except NameError:
        print("\n[Loading data from CSV files]")
        MIS_PROTEINAS, MIS_REACCIONES = load_data_from_csv()

    # ========== Preparar datos ==========
    print(f"\n{'Proteínas:':40s} {len(MIS_PROTEINAS)}")
    for nombre, seq in MIS_PROTEINAS.items():
        print(f"  • {nombre:30s} {len(seq)} aa")

    print(f"\n{'Reacciones:':40s} {len(MIS_REACCIONES)}")
    for nombre, rxn in MIS_REACCIONES.items():
        print(f"  • {nombre:30s} {rxn[:40]}...")

    # ========== Cargar modelos ==========
    print("\n" + "-" * 70)
    print("Cargando modelos...")
    print("-" * 70)

    # Protein encoder
    protein_encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",
        pooling="attention",
        proj_dim=256
    ).to(device).eval()
    print("  ✓ Protein encoder listo")

    # Reaction encoder (necesitamos dimensiones de una reacción de ejemplo)
    rxn_ejemplo = list(MIS_REACCIONES.values())[0]
    reacts, prods = parse_reaction_smiles(rxn_ejemplo)
    data_ejemplo = build_transition_graph(reacts, prods, use_enhanced_features=True)

    reaction_encoder = ReactionGNNEnhanced(
        x_dim=data_ejemplo.x.size(1),
        e_dim=data_ejemplo.edge_attr.size(1),
        hidden=128,
        layers=3,
        out_dim=256
    ).to(device).eval()
    print("  ✓ Reaction encoder listo")

    # ========== Codificar proteínas ==========
    print("\n" + "-" * 70)
    print("Codificando proteínas...")
    print("-" * 70)

    protein_names = list(MIS_PROTEINAS.keys())
    protein_seqs = list(MIS_PROTEINAS.values())

    batch_prot = protein_encoder.tokenize(protein_seqs, max_len=1024)
    batch_prot = {k: v.to(device) for k, v in batch_prot.items()}

    with torch.no_grad():
        protein_embeddings = protein_encoder(batch_prot)

    print(f"  ✓ {len(protein_names)} proteínas codificadas")
    print(f"    Shape: {protein_embeddings.shape}")

    # ========== Codificar reacciones ==========
    print("\n" + "-" * 70)
    print("Codificando reacciones...")
    print("-" * 70)

    reaction_names = list(MIS_REACCIONES.keys())
    reaction_smiles = list(MIS_REACCIONES.values())

    reaction_embeddings_list = []
    for rxn_smiles in reaction_smiles:
        reacts, prods = parse_reaction_smiles(rxn_smiles)
        data = build_transition_graph(reacts, prods, use_enhanced_features=True)
        data = data.to(device)

        with torch.no_grad():
            z = reaction_encoder(data)
            reaction_embeddings_list.append(z)

    reaction_embeddings = torch.cat(reaction_embeddings_list, dim=0)

    print(f"  ✓ {len(reaction_names)} reacciones codificadas")
    print(f"    Shape: {reaction_embeddings.shape}")

    # ========== Matriz de similitud ==========
    print("\n" + "=" * 70)
    print("RESULTADOS: Matriz de Similitud Proteína-Reacción")
    print("=" * 70)
    print("\n(Valores más altos = mejor match)")
    print()

    similarity_matrix = protein_embeddings @ reaction_embeddings.t()

    # Header
    print(" " * 35, end="")
    for rxn_name in reaction_names:
        print(f"{rxn_name[:15]:17s}", end="")
    print()
    print("-" * (35 + 17 * len(reaction_names)))

    # Filas
    for i, prot_name in enumerate(protein_names):
        print(f"{prot_name[:33]:35s}", end="")
        for j in range(len(reaction_names)):
            sim = similarity_matrix[i, j].item()
            print(f"{sim:17.4f}", end="")
        print()

    # ========== Top matches ==========
    print("\n" + "=" * 70)
    print("TOP MATCHES (para cada reacción)")
    print("=" * 70)

    for j, rxn_name in enumerate(reaction_names):
        scores = similarity_matrix[:, j]
        top_idx = torch.argmax(scores).item()
        top_score = scores[top_idx].item()

        print(f"\n{rxn_name}:")
        print(f"  → Mejor match: {protein_names[top_idx]}")
        print(f"  → Score: {top_score:.4f}")

        # Mostrar top 3
        top3_scores, top3_indices = torch.topk(scores, k=min(3, len(protein_names)))
        print(f"  → Top 3:")
        for rank, (idx, score) in enumerate(zip(top3_indices, top3_scores), 1):
            print(f"       {rank}. {protein_names[idx]:30s} {score.item():.4f}")

    # ========== Resumen ==========
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"""
Estadísticas:
  • Similitud promedio:  {similarity_matrix.mean().item():.4f}
  • Similitud máxima:    {similarity_matrix.max().item():.4f}
  • Similitud mínima:    {similarity_matrix.min().item():.4f}

Nota: Los modelos NO están entrenados, así que los scores son aleatorios!
      Después de entrenar con CLIP loss, los matches correctos tendrían
      scores mucho más altos (~0.7-0.9) que los incorrectos (~0.1-0.3).

Para mejorar los resultados:
  1. Entrenar los modelos con pares conocidos enzima-reacción
  2. Usar el CLIP loss para alinear embeddings
  3. Iterar sobre un dataset de 10K+ pares

¡Pero la arquitectura ya funciona! Solo falta entrenar.
""")

    print("=" * 70)


if __name__ == "__main__":
    print("\n")
    print("#" * 70)
    print("# PLAYGROUND: Prueba tus Proteínas y Reacciones")
    print("#" * 70)
    print()
    print("INSTRUCCIONES:")
    print("  OPCIÓN 1 (Recomendado): Editar CSVs")
    print("    - Edita data/proteins.csv para agregar/cambiar proteínas")
    print("    - Edita data/reactions_extended.csv para agregar/cambiar reacciones")
    print("  OPCIÓN 2: Usar datos inline")
    print("    - Abre test_playground.py")
    print("    - Descomenta las secciones MIS_PROTEINAS y MIS_REACCIONES")
    print("    - Cambia los datos ahí")
    print()
    print("Ejecutando con datos cargados...")
    print()

    try:
        playground()

        print("\n" + "#" * 70)
        print("# PLAYGROUND COMPLETADO!")
        print("#" * 70)
        print()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
