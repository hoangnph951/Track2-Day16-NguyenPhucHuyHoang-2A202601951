import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LIGHTGBM CREDIT CARD FRAUD BENCHMARK")
print("=" * 60)

print(f"\nDataset: {DATA_PATH}")

load_start = time.perf_counter()

df = pd.read_csv(DATA_PATH)

load_end = time.perf_counter()
load_time = load_end - load_start

print(f"Dataset shape: {df.shape}")
print(f"Load time: {load_time:.4f} seconds")


# ============================================================
# SPLIT FEATURES / LABEL
# ============================================================

X = df.drop(columns=["Class"])
y = df["Class"]

print(f"\nNormal transactions: {(y == 0).sum()}")
print(f"Fraud transactions : {(y == 1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 2. TRAIN LIGHTGBM MODEL
# ============================================================

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

print("\nTraining LightGBM...")

train_start = time.perf_counter()

model.fit(X_train, y_train)

train_end = time.perf_counter()
training_time = train_end - train_start

print(f"Training time: {training_time:.4f} seconds")


# ============================================================
# 3. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc_roc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)


# ============================================================
# 4. INFERENCE LATENCY - SINGLE ROW
# ============================================================

single_sample = X_test.iloc[[0]]

# Warm-up để lần đo đầu không bị ảnh hưởng bởi initialization
for _ in range(10):
    model.predict_proba(single_sample)

# Đo nhiều lần rồi lấy trung bình để kết quả ổn định hơn
latencies = []

for _ in range(100):
    start = time.perf_counter()
    model.predict_proba(single_sample)
    end = time.perf_counter()

    latencies.append((end - start) * 1000)  # milliseconds

inference_latency_ms = float(np.mean(latencies))


# ============================================================
# 5. INFERENCE THROUGHPUT - 1000 ROWS
# ============================================================

num_inference_samples = min(1000, len(X_test))
batch = X_test.iloc[:num_inference_samples]

# Warm-up
model.predict_proba(batch)

throughput_start = time.perf_counter()

model.predict_proba(batch)

throughput_end = time.perf_counter()

batch_inference_time = throughput_end - throughput_start

inference_throughput = (
    num_inference_samples / batch_inference_time
)


# ============================================================
# 6. RESULTS
# ============================================================

results = {
    "dataset": {
        "path": str(DATA_PATH),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": int(X.shape[1]),
    },

    "timing": {
        "data_load_seconds": float(load_time),
        "training_seconds": float(training_time),
    },

    "metrics": {
        "auc_roc": float(auc_roc),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    },

    "inference": {
        "single_row_latency_ms": float(inference_latency_ms),
        "batch_size": int(num_inference_samples),
        "batch_inference_seconds": float(batch_inference_time),
        "throughput_samples_per_second": float(inference_throughput),
    },
}


# ============================================================
# PRINT RESULT
# ============================================================

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(f"""
Data load time        : {load_time:.4f} s
Training time         : {training_time:.4f} s

AUC-ROC               : {auc_roc:.6f}
Accuracy              : {accuracy:.6f}
F1-Score              : {f1:.6f}
Precision             : {precision:.6f}
Recall                : {recall:.6f}

Inference latency     : {inference_latency_ms:.4f} ms / sample
Inference throughput  : {inference_throughput:.2f} samples/s
""")


# ============================================================
# SAVE JSON
# ============================================================

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to: {RESULT_PATH}")