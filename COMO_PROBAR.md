# 🧪 Cómo Probar los Encoders - Guía Rápida

Esta guía te muestra exactamente cómo probar los encoders de proteínas y reacciones con tus propios datos.

---

## 🚀 Scripts Disponibles

### 1. **test_protein_simple.py** - Prueba el encoder de proteínas

```bash
cd C:\Users\Latitude 7390\desktop\project
python scripts/test_protein_simple.py
```

**Qué hace:**
- ✓ Codifica 2 proteínas cortas
- ✓ Compara múltiples secuencias
- ✓ Maneja secuencias largas (>1000 aa)
- ✓ Muestra matriz de similitud

**Personalizar:**
Abre el archivo y cambia la línea 50:
```python
mi_secuencia = "MSKGEELF..."  # PON TU SECUENCIA AQUÍ
```

---

### 2. **test_reaction_simple.py** - Prueba el encoder de reacciones

```bash
python scripts/test_reaction_simple.py
```

**Qué hace:**
- ✓ Codifica reacciones SMILES
- ✓ Compara múltiples reacciones
- ✓ Analiza qué enlaces cambian
- ✓ Muestra similitudes

**Personalizar:**
Abre el archivo y cambia la línea 54:
```python
mi_reaccion = "[C:1]#[C:2]>>[C:1]=[C:2]"  # TU REACCIÓN AQUÍ
```

**⚠️ IMPORTANTE:** Las reacciones DEBEN tener mapeo de átomos: `[C:1]`, `[N:2]`, etc.

---

### 3. **test_playground.py** - Prueba proteínas + reacciones juntas

```bash
python scripts/test_playground.py
```

**Qué hace:**
- ✓ Codifica TUS proteínas y reacciones
- ✓ Calcula matriz de similitud proteína-reacción
- ✓ Encuentra el mejor match para cada reacción
- ✓ Muestra top 3 matches

**Personalizar:**
Abre el archivo y cambia estas secciones (líneas 17-29):

```python
# Tus proteínas
MIS_PROTEINAS = {
    "Mi enzima 1": "MSKGEELF...",  # PON TUS SECUENCIAS
    "Mi enzima 2": "MAHHHHH...",
}

# Tus reacciones
MIS_REACCIONES = {
    "Reacción A": "[N:1]=[N:2]>>[N:1][N:2]",  # PON TUS REACCIONES
    "Reacción B": "[C:1]=[C:2]>>[C:1][C:2]",
}
```

Luego ejecuta de nuevo!

---

## 📊 Ejemplo de Salida

### Protein Encoder

```
Secuencia 1: 45 aminoácidos
Secuencia 2: 30 aminoácidos

Resultado:
  Shape: torch.Size([2, 256])
  Norma embedding 1: 1.0000
  Norma embedding 2: 1.0000

Similitud coseno entre las dos: 0.8234
```

### Reaction Encoder

```
Reacción 1: [N:1]=[N:2]>>[N:1][N:2]
Reacción 2: [C:1]=[C:2]>>[C:1][C:2]

Embedding 1: torch.Size([1, 256]), norma=1.0000
Embedding 2: torch.Size([1, 256]), norma=1.0000

Similitud coseno: 0.9123
(Son similares porque ambas son reducciones!)
```

### Playground - Matriz de Similitud

```
RESULTADOS: Matriz de Similitud Proteína-Reacción
(Valores más altos = mejor match)

                                   Reacción A       Reacción B       Reacción C
------------------------------------------------------------------------------------
Mi enzima 1                        0.0856           0.0923           0.0784
Mi enzima 2                        0.0912           0.0867           0.0891
Mi enzima 3                        0.0789           0.0845           0.0923

TOP MATCHES (para cada reacción)

Reacción A:
  → Mejor match: Mi enzima 2
  → Score: 0.0912
```

---

## 🎯 Casos de Uso Prácticos

### Caso 1: Comparar dos proteínas mías

```python
# En test_protein_simple.py, línea 64
secuencias = {
    "Proteína salvaje": "MSKGEELF...",
    "Mutante K42A": "MSKGEELA...",  # Cambio en posición 42
}
```

Ejecuta y ve qué tan similares son (debería ser ~0.95+).

### Caso 2: Comparar reacciones similares

```python
# En test_reaction_simple.py, línea 69
reacciones = {
    "Hidrogenación 1": "[C:1]=[C:2]>>[C:1][C:2]",
    "Hidrogenación 2": "[C:1]=[C:2].[H:3][H:4]>>[C:1]([H:3])[C:2]([H:4])",
}
```

Ejecuta y ve las similitudes.

### Caso 3: Encontrar enzima para mi reacción

```python
# En test_playground.py
MIS_PROTEINAS = {
    "Reductasa A": "MTEQSKLVNIDPK...",
    "Oxidasa B": "MKKILAVAAALA...",
    "Hidrolasa C": "MASSKSTVVAGLL...",
}

MIS_REACCIONES = {
    "Mi reacción de interés": "[N:1]=[N:2]>>[N:1][N:2]",
}
```

Ejecuta y ve cuál enzima tiene el score más alto.

---

## 🔧 Tips y Trucos

### 1. Secuencias de Proteínas

✅ **Correcto:**
```python
seq = "MSKGEELFTGVVPILVELDGDV"
```

❌ **Incorrecto:**
```python
seq = "MSK GEE LFT"  # No espacios
seq = "mskgeelf"      # Mayúsculas solamente
seq = "MSK123"        # Solo letras AA válidas
```

**Caracteres válidos:**
- A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- X (desconocido), U (selenocisteína), O (pirrolisina)
- B, Z, J (ambiguos, pero aceptados)

### 2. Reacciones SMILES

✅ **Correcto (CON mapeo):**
```python
rxn = "[C:1]=[O:2]>>[C:1][O:2]"
rxn = "[N:1]=[N:2].[H:3][H:4]>>[N:1][N:2].[H:3][H:4]"
```

❌ **Incorrecto (SIN mapeo):**
```python
rxn = "C=O>>CO"  # Falta :1, :2, etc.
```

**Cómo agregar mapeo:**
- Manualmente: numera cada átomo
- Automáticamente: usa RXNMapper (no incluido aquí)

### 3. Interpretando Similitudes

**Sin entrenar (modelos con pesos aleatorios):**
- Similitudes: 0.00 - 0.30 (aleatorias)
- No hay patrón real

**Después de entrenar:**
- Matches correctos: 0.70 - 0.95
- Matches incorrectos: 0.05 - 0.30
- La diagonal de la matriz sería alta

### 4. Velocidad

**En CPU:**
- Proteína corta (50 aa): ~3 segundos
- Proteína larga (500 aa): ~8 segundos
- Reacción simple: <1 segundo

**En GPU (si disponible):**
- Proteína: ~0.5 segundos
- Reacción: <0.1 segundos

Cambia `device="cpu"` a `device="cuda"` en los scripts.

---

## 🐛 Problemas Comunes

### Error: "Bad SMILES in reaction"

**Causa:** Reacción sin mapeo de átomos o SMILES inválido.

**Solución:**
```python
# Mal:
rxn = "C=O>>CO"

# Bien:
rxn = "[C:1]=[O:2]>>[C:1][O:2]"
```

### Error: "Invalid characters found"

**Causa:** Secuencia contiene caracteres no-aminoácidos.

**Solución:**
```python
# Mal:
seq = "MSK123GEE"

# Bien:
seq = "MSKGEE"
```

### Error: Out of Memory

**Causa:** Secuencia muy larga o modelo muy grande.

**Solución:**
```python
# Usa modelo pequeño
plm_name="facebook/esm2_t12_35M_UR50D"  # 35M params

# O procesa en chunks
from protein_encoder.utils import encode_long_sequence
```

### Advertencia: "Some weights not initialized"

**No es un error!** Es normal. Los pesos del "pooler" de ESM2 no se usan.
Puedes ignorar este warning.

---

## 📚 Ejemplos de Secuencias Reales

### Proteínas Conocidas

```python
# GFP (Green Fluorescent Protein)
gfp = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# Insulina humana (cadena B)
insulin = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"

# Lisozima (primera parte)
lysozyme = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
```

### Reacciones Comunes (con mapeo)

```python
# Hidrogenación de alqueno
alkene_h2 = "[C:1]=[C:2].[H:3][H:4]>>[C:1]([H:3])[C:2]([H:4])"

# Reducción de carbonilo
carbonyl_red = "[C:1]=[O:2].[H:3][H:4]>>[C:1][O:2].[H:3][H:4]"

# Hidrólisis de éster
ester_hydro = "[C:1](=[O:2])[O:3][C:4].[H:5][O:6][H:7]>>[C:1](=[O:2])[O:5][H:7].[O:3]([C:4])[H:6]"

# Reducción de nitrilo
nitrile_red = "[C:1]#[N:2]>>[C:1]=[N:2]"
```

---

## 🎮 Tutorial Paso a Paso

### Tutorial 1: Mi Primera Prueba

1. Abre la terminal
2. Ve al proyecto:
   ```bash
   cd C:\Users\Latitude 7390\desktop\project
   ```
3. Ejecuta el test más simple:
   ```bash
   python scripts/test_protein_simple.py
   ```
4. Observa los resultados. ¿Los embeddings tienen norma 1.0? ✓
5. ¿Las similitudes están entre -1 y 1? ✓

### Tutorial 2: Probar Mi Secuencia

1. Abre `scripts/test_protein_simple.py` en un editor
2. Ve a la línea 50 (función `test_tu_secuencia()`)
3. Cambia `mi_secuencia = "..."` por tu secuencia
4. Guarda el archivo
5. Ejecuta:
   ```bash
   python scripts/test_protein_simple.py
   ```
6. Ve los resultados para TU secuencia!

### Tutorial 3: Comparar Mis Proteínas

1. Abre `scripts/test_protein_simple.py`
2. Ve a la línea 78 (función `test_multiples()`)
3. Cambia el diccionario `secuencias = {...}`:
   ```python
   secuencias = {
       "Mi proteína 1": "MSKGEELF...",
       "Mi proteína 2": "MAHHHHH...",
       "Mi proteína 3": "MALWMRLL...",
   }
   ```
4. Ejecuta y ve la matriz de similitud!

### Tutorial 4: Matching Completo

1. Abre `scripts/test_playground.py`
2. Cambia `MIS_PROTEINAS` con tus enzimas
3. Cambia `MIS_REACCIONES` con tus reacciones
4. Ejecuta:
   ```bash
   python scripts/test_playground.py
   ```
5. Ve qué enzima matchea mejor con cada reacción!

---

## 💡 Ideas para Experimentar

1. **Mutaciones:** Cambia 1 aminoácido, ve cómo cambia la similitud
2. **Familias:** Compara enzimas de la misma familia (debería ser >0.8)
3. **Reacciones:** Compara hidrogenaciones vs oxidaciones (debería ser <0.5)
4. **Longitud:** Prueba secuencias de 50, 500, 1500 aa
5. **Subsecuencias:** Extrae el dominio activo, compara con la proteína completa

---

## 📞 ¿Necesitas Ayuda?

Si algo no funciona:

1. **Verifica la instalación:**
   ```bash
   python -c "import protein_encoder; import reaction_encoder; print('OK!')"
   ```

2. **Verifica las dependencias:**
   ```bash
   pip list | grep -E "torch|transformers|rdkit"
   ```

3. **Mira los errores:** Lee el traceback completo

4. **Prueba el modelo pequeño:** Cambia a `esm2_t12_35M_UR50D`

---

## 🎯 Siguiente Nivel

Una vez que domines las pruebas básicas:

1. Lee `PROTEIN_ENCODER_README.md` para detalles técnicos
2. Lee `IMPROVEMENTS.md` para el reaction encoder
3. Explora `demo_clipzyme_complete.py` para ver la integración completa
4. Considera implementar el training loop (ver TODO abajo)

---

## ✅ Checklist Rápido

Antes de probar, verifica:

- [ ] Estás en el directorio del proyecto
- [ ] Tienes instalado: `transformers`, `rdkit`, `torch`
- [ ] Tus secuencias son MAYÚSCULAS y solo contienen AA válidos
- [ ] Tus reacciones tienen mapeo de átomos `:1`, `:2`, etc.
- [ ] Sabes que los modelos NO están entrenados (similitudes aleatorias)

---

## 🚀 Resumen de Comandos

```bash
# Ir al proyecto
cd C:\Users\Latitude 7390\desktop\project

# Probar proteínas
python scripts/test_protein_simple.py

# Probar reacciones
python scripts/test_reaction_simple.py

# Playground (proteínas + reacciones)
python scripts/test_playground.py

# Demo completo (con enzimas reales)
python scripts/demo_clipzyme_complete.py
```

---

**¡Listo! Ahora tienes todo lo necesario para probar los encoders con tus propios datos.** 🎉
