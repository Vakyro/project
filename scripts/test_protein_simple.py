"""
Script simple para probar el encoder de proteínas.
Usa tus propias secuencias y ve los resultados.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import csv
from protein_encoder import ProteinEncoderESM2


def test_basic():
    """Prueba básica - codifica 2 proteínas y compara."""
    print("=" * 60)
    print("TEST 1: Prueba Básica")
    print("=" * 60)

    # Inicializar encoder (modelo pequeño para ser rápido)
    print("\nCargando modelo...")
    encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",  # Modelo pequeño
        pooling="attention",
        proj_dim=256
    ).eval()

    # Load sequences from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'proteins.csv')
    sequences_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['length']) <= 50:
                sequences_data.append((row['sequence'], row['name']))
                if len(sequences_data) >= 2:
                    break

    seq1, name1 = sequences_data[0]
    seq2, name2 = sequences_data[1]

    print(f"\nSecuencia 1 ({name1}): {len(seq1)} aminoácidos")
    print(f"Secuencia 2 ({name2}): {len(seq2)} aminoácidos")

    # Codificar
    print("\nCodificando...")
    embeddings = encoder.encode([seq1, seq2], device="cpu")

    print(f"\nResultado:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Norma embedding 1: {torch.norm(embeddings[0]).item():.4f}")
    print(f"  Norma embedding 2: {torch.norm(embeddings[1]).item():.4f}")

    # Similitud
    similarity = (embeddings[0] @ embeddings[1]).item()
    print(f"\nSimilitud coseno entre las dos: {similarity:.4f}")
    print(f"  (1.0 = idénticas, 0.0 = ortogonales, -1.0 = opuestas)")

    print("\n✓ Prueba completada!")


def test_tu_secuencia():
    """Codifica una secuencia que tú proporciones."""
    print("\n" + "=" * 60)
    print("TEST 2: Codifica TU Secuencia")
    print("=" * 60)

    # Load a full GFP sequence from CSV (users can edit CSV to add their own)
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'proteins.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'GFP-full' in row['name']:
                mi_secuencia = row['sequence']
                protein_name = row['name']
                break

    print(f"\nTu secuencia ({protein_name}): {len(mi_secuencia)} aminoácidos")
    print(f"Primeros 50 aa: {mi_secuencia[:50]}...")

    # Cargar modelo
    encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",
        pooling="attention",
        proj_dim=256
    ).eval()

    # Codificar
    print("\nCodificando...")
    embedding = encoder.encode([mi_secuencia], device="cpu")

    print(f"\nEmbedding generado:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Norma: {torch.norm(embedding).item():.4f} (debe ser ~1.0)")
    print(f"  Primeros 10 valores: {embedding[0, :10].tolist()}")

    print("\n✓ Tu secuencia fue codificada exitosamente!")


def test_multiples():
    """Codifica varias secuencias y ve cuáles son más similares."""
    print("\n" + "=" * 60)
    print("TEST 3: Múltiples Secuencias - ¿Cuáles son similares?")
    print("=" * 60)

    # Load sequences from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'proteins.csv')
    secuencias = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['length']) <= 60:
                secuencias[row['name']] = row['sequence']
                if len(secuencias) >= 4:
                    break

    nombres = list(secuencias.keys())
    seqs = list(secuencias.values())

    print(f"\nSecuencias a comparar (from CSV):")
    for nombre, seq in secuencias.items():
        print(f"  {nombre}: {len(seq)} aa")

    # Cargar y codificar
    encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",
        pooling="attention",
        proj_dim=256
    ).eval()

    print("\nCodificando todas...")
    embeddings = encoder.encode(seqs, device="cpu")

    # Matriz de similitud
    print("\nMatriz de Similitud:")
    print("(1.0 = idénticas, 0.0 = no relacionadas)")
    print()

    # Header
    print(" " * 12, end="")
    for nombre in nombres:
        print(f"{nombre:12s}", end="")
    print()

    # Filas
    sim_matrix = embeddings @ embeddings.t()
    for i, nombre_i in enumerate(nombres):
        print(f"{nombre_i:12s}", end="")
        for j in range(len(nombres)):
            sim = sim_matrix[i, j].item()
            print(f"{sim:12.4f}", end="")
        print()

    # Encuentra las más similares (excluyendo diagonal)
    print("\nPares más similares:")
    for i in range(len(nombres)):
        for j in range(i+1, len(nombres)):
            sim = sim_matrix[i, j].item()
            print(f"  {nombres[i]} <-> {nombres[j]}: {sim:.4f}")

    print("\n✓ Comparación completada!")


def test_larga():
    """Prueba con una secuencia larga (>1000 aa)."""
    print("\n" + "=" * 60)
    print("TEST 4: Secuencia Larga (chunking automático)")
    print("=" * 60)

    # Create a long sequence by repeating a base from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'proteins.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'GFP-full' in row['name']:
                base = row['sequence']
                break

    seq_larga = base * 8  # ~2000 aminoácidos

    print(f"\nSecuencia larga (8x GFP): {len(seq_larga)} aminoácidos")
    print("(Esto excede el límite de 1024 tokens de ESM2)")

    encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",
        pooling="attention",
        proj_dim=256
    ).eval()

    # El chunking es automático si usas encode_long_sequence
    from protein_encoder.utils import encode_long_sequence

    print("\nCodificando con chunking automático...")
    embedding = encode_long_sequence(
        encoder,
        seq_larga,
        device="cpu",
        max_len=1000,
        overlap=50
    )

    print(f"\nResultado:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Norma: {torch.norm(embedding).item():.4f}")

    print("\n✓ Secuencia larga procesada exitosamente!")
    print("  (Se dividió en chunks, se codificó cada uno, y se promedió)")


if __name__ == "__main__":
    print("\n")
    print("#" * 60)
    print("# PRUEBAS DEL PROTEIN ENCODER")
    print("#" * 60)

    try:
        # Ejecutar todas las pruebas
        test_basic()
        test_tu_secuencia()
        test_multiples()
        test_larga()

        print("\n" + "#" * 60)
        print("# ¡TODAS LAS PRUEBAS PASARON!")
        print("#" * 60)
        print("\nAhora puedes:")
        print("  1. Cambiar las secuencias en este script")
        print("  2. Agregar tus propias secuencias")
        print("  3. Comparar proteínas que te interesen")
        print()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
