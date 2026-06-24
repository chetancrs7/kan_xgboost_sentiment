# DeBERTa-v3 Embedding Generation Pipeline

## Overview

This module is responsible for generating high-quality semantic text embeddings using **DeBERTa-v3-base**. The generated embeddings serve as the common feature representation for all downstream models in this project, including:

* KAN (Kolmogorov-Arnold Network)
* XGBoost
* Parallel Adaptive Hybrid Ensemble (KAN + XGBoost)

The purpose of this design is to ensure that every model receives **exactly the same input features**, making model comparisons fair, reproducible, and scientifically valid.

---

# File Structure

```text
src/
└── embeddings/
    ├── config.py
    ├── embedding_generator.py
    └── generate_embeddings.py
```

---

# Overall Pipeline

The complete embedding generation workflow is shown below:

```text
train.csv
val.csv
test.csv
      │
      ▼
generate_embeddings.py
      │
      ▼
config.py
      │
      ▼
embedding_generator.py
      │
      ▼
DeBERTa-v3-base
      │
      ▼
Mean Pooling
      │
      ▼
768-Dimensional Sentence Embeddings
      │
      ▼
train_embeddings.npy
val_embeddings.npy
test_embeddings.npy

train_labels.npy
val_labels.npy
test_labels.npy
```

These `.npy` files become the shared input for all subsequent machine learning models.

---

# File 1: config.py

## Purpose

`config.py` acts as the central configuration module for the embedding generation pipeline.

Instead of hardcoding settings throughout multiple scripts, all configurable parameters are stored in a single location. This ensures consistency across experiments and simplifies maintenance.

---

## Responsibilities

### 1. Store Default Configuration Values

The file defines all default parameters used by the pipeline:

```python
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128
DEFAULT_SEED = 42
```

These values control:

* Which transformer model is used
* Batch size during inference
* Maximum token length
* Reproducibility settings

---

### 2. Define Dataset Locations

```python
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "splits"
```

Expected structure:

```text
data/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

### 3. Define Output Locations

```python
DEFAULT_OUTPUT_DIR =
    PROJECT_ROOT / "data" / "embeddings" / "deberta_v3_base"
```

Generated files will be stored here:

```text
data/
└── embeddings/
    └── deberta_v3_base/
        ├── train_embeddings.npy
        ├── train_labels.npy
        ├── val_embeddings.npy
        ├── val_labels.npy
        ├── test_embeddings.npy
        └── test_labels.npy
```

---

### 4. Provide Reproducibility

The `set_seed()` function ensures deterministic results.

It seeds:

* Python random module
* NumPy
* PyTorch
* CUDA

This guarantees that the same inputs produce identical outputs across runs.

---

### 5. Configure Logging

The `configure_logging()` function standardizes log messages throughout the pipeline.

Example:

```text
12:30:15 | INFO | embeddings | Loading model...
```

---

## Main Component: EmbeddingConfig

`EmbeddingConfig` is an immutable configuration object that stores all settings required for a single embedding generation run.

Example:

```python
config = EmbeddingConfig(
    batch_size=64,
    max_length=160,
    device="cuda"
)
```

This object is later passed to all other pipeline components.

---

## config.py Pipeline

```text
Default Parameters
       │
       ▼
EmbeddingConfig
       │
       ▼
Path Management
Seed Management
Logging Configuration
       │
       ▼
Shared Configuration Object
```

---

# File 2: embedding_generator.py

## Purpose

`embedding_generator.py` contains the actual embedding engine.

Its responsibility is to convert raw text into fixed-length numerical vectors using DeBERTa-v3-base.

---

## Input

Example:

```python
[
    "I love this movie",
    "This product is terrible"
]
```

---

## Output

```python
array(
    shape=(2, 768)
)
```

Each text is transformed into a 768-dimensional embedding vector.

---

## Main Component: EmbeddingGenerator

The `EmbeddingGenerator` class performs the complete embedding process.

---

### Step 1: Load DeBERTa-v3

The constructor loads:

```python
AutoTokenizer
AutoModel
```

using:

```python
microsoft/deberta-v3-base
```

---

### Step 2: Device Selection

The generator automatically selects:

```text
GPU (CUDA)
```

if available, otherwise:

```text
CPU
```

The user can also manually force a device.

---

### Step 3: Tokenization

Raw text is converted into token IDs.

Example:

```text
"I love this movie"
```

becomes:

```text
[CLS] I love this movie [SEP]
```

and then numerical token IDs.

---

### Step 4: Transformer Encoding

The tokenized input is passed through DeBERTa-v3.

Output:

```text
Token-Level Embeddings
```

Shape:

```python
(batch_size,
 sequence_length,
 768)
```

Each token receives its own 768-dimensional vector.

---

### Step 5: Mean Pooling

Since downstream models require one vector per sentence, token embeddings are aggregated using masked mean pooling.

This process:

1. Ignores padding tokens
2. Averages valid token vectors
3. Produces a single sentence representation

Result:

```python
(batch_size, 768)
```

---

### Step 6: Return Embeddings

The embeddings are returned as NumPy arrays:

```python
numpy.ndarray
```

with dtype:

```python
float32
```

---

## embedding_generator.py Pipeline

```text
Raw Text
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
DeBERTa-v3 Encoder
    │
    ▼
Token Embeddings
    │
    ▼
Masked Mean Pooling
    │
    ▼
Sentence Embeddings
    │
    ▼
NumPy Array (n,768)
```

---

# File 3: generate_embeddings.py

## Purpose

`generate_embeddings.py` is the execution script.

This is the file users run to generate embeddings for all dataset splits.

Example:

```bash
python src/embeddings/generate_embeddings.py
```

---

## Responsibilities

### 1. Parse User Arguments

The script accepts optional command-line parameters:

```bash
--batch-size
--max-length
--device
--output-dir
--model-name
```

Example:

```bash
python generate_embeddings.py --batch-size 64 --device cuda
```

---

### 2. Create Configuration Object

Arguments are converted into:

```python
EmbeddingConfig
```

which becomes the shared configuration throughout the pipeline.

---

### 3. Create Output Directory

The script automatically creates the output folder if it does not already exist.

---

### 4. Load Dataset Splits

For each split:

```text
train
val
test
```

the script:

* Reads the CSV file
* Validates required columns
* Removes null rows
* Removes empty text rows

---

### 5. Generate Embeddings

The cleaned text is passed to:

```python
EmbeddingGenerator.generate()
```

which returns:

```python
(n_samples,768)
```

embeddings.

---

### 6. Extract Labels

Labels are extracted directly from the dataset:

```python
label
```

column.

---

### 7. Save Outputs

The script saves:

```text
train_embeddings.npy
train_labels.npy

val_embeddings.npy
val_labels.npy

test_embeddings.npy
test_labels.npy
```

---

## generate_embeddings.py Pipeline

```text
train.csv
val.csv
test.csv
      │
      ▼
Load Data
      │
      ▼
Validate Columns
      │
      ▼
Remove Null/Empty Rows
      │
      ▼
EmbeddingGenerator.generate()
      │
      ▼
768-D Embeddings
      │
      ▼
Save .npy Files
```

---

# How The Three Files Work Together

The three files follow a separation-of-concerns architecture.

Each file has one clearly defined responsibility.

---

## Step 1: Configuration

`config.py`

Provides:

* Paths
* Hyperparameters
* Random seeds
* Logging configuration

Output:

```python
EmbeddingConfig
```

---

## Step 2: Embedding Generation Engine

`embedding_generator.py`

Uses:

```python
EmbeddingConfig
```

to:

* Load DeBERTa-v3
* Encode text
* Generate embeddings

Output:

```python
numpy.ndarray
```

containing sentence embeddings.

---

## Step 3: Pipeline Execution

`generate_embeddings.py`

Coordinates the complete workflow:

1. Loads train/validation/test data
2. Creates EmbeddingGenerator
3. Generates embeddings
4. Saves embeddings and labels

Output:

```text
train_embeddings.npy
val_embeddings.npy
test_embeddings.npy

train_labels.npy
val_labels.npy
test_labels.npy
```

---

# Complete System Architecture

```text
                    config.py
                         │
                         ▼
                EmbeddingConfig
                         │
                         ▼
             embedding_generator.py
                         │
                         ▼
                 EmbeddingGenerator
                         │
                         ▼
             generate_embeddings.py
                         │
                         ▼
              train.csv / val.csv / test.csv
                         │
                         ▼
                 DeBERTa-v3-base
                         │
                         ▼
                Mean-Pooled Embeddings
                         │
                         ▼
     train_embeddings.npy
     val_embeddings.npy
     test_embeddings.npy
                         │
                         ▼
        KAN / XGBoost / Hybrid Ensemble
```

---

# Why This Architecture Was Chosen

This architecture provides several advantages:

### Reusability

The embedding generator can be reused by multiple models without modification.

### Reproducibility

All experiments use identical configuration settings and random seeds.

### Maintainability

Configuration, embedding logic, and execution logic are separated into independent modules.

### Fair Model Comparison

KAN, XGBoost, and the Hybrid Ensemble all receive exactly the same feature representations, ensuring that performance differences are caused by the models themselves rather than differences in preprocessing or feature extraction.

This makes the experimental results more reliable and scientifically valid.
