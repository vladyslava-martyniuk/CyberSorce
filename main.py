import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

folder = Path.cwd()

possible_files = [
    "cybersecurity_network_traffic_cleaned.csv",
    "cybersecurity_network_traffic.csv",
    "cybersecurity_network_traffic_synthetic.csv"
]

data_path = next(
    (folder / file for file in possible_files if (folder / file).exists()),
    None
)

if data_path is None:
    raise FileNotFoundError(
        "Не знайдено CSV-файл датасету в поточній папці."
    )

df = pd.read_csv(data_path)

df.columns = df.columns.str.strip()

target_column = next(
    (col for col in df.columns if col.lower() == "label"),
    None
)

if target_column is None:
    raise ValueError("У датасеті не знайдено стовпець Label.")

df = df.dropna(subset=[target_column]).drop_duplicates().copy()

df[target_column] = (
    df[target_column]
    .astype(str)
    .str.strip()
    .str.upper()
)

df[target_column] = df[target_column].replace({
    "BENIGN": 0,
    "ATTACK": 1
})

if not set(df[target_column].unique()).issubset({0, 1}):
    raise ValueError(
        "У Label мають бути лише значення BENIGN і ATTACK."
    )

df[target_column] = df[target_column].astype(int)

if df[target_column].nunique() < 2:
    raise ValueError(
        "Для навчання потрібні обидва класи: BENIGN та ATTACK."
    )

if {"Total Bytes Sent", "Total Bytes Received"}.issubset(df.columns):
    df["total_bytes"] = (
        df["Total Bytes Sent"] + df["Total Bytes Received"]
    )

if {"Total Fwd Packets", "Total Backward Packets"}.issubset(df.columns):
    df["total_packets"] = (
        df["Total Fwd Packets"] + df["Total Backward Packets"]
    )

if {"total_bytes", "duration"}.issubset(df.columns):
    df["bytes_per_sec"] = (
        df["total_bytes"] / df["duration"].replace(0, np.nan)
    )

if {"total_packets", "duration"}.issubset(df.columns):
    df["packets_per_sec"] = (
        df["total_packets"] / df["duration"].replace(0, np.nan)
    )

X = df.drop(columns=[target_column])
y = df[target_column]

X = X.replace([np.inf, -np.inf], np.nan)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

transformers = []

if numeric_features:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "k-NN": KNeighborsClassifier(
        n_neighbors=5
    )
}

results = []
confusion_matrices = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    confusion_matrices[name] = cm

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Модель": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["FN", "Recall", "F1-Score"],
    ascending=[True, False, False]
)

print("РЕЗУЛЬТАТИ ОЦІНЮВАННЯ МОДЕЛЕЙ")

display(
    results_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}"
    })
)

results_df.to_csv(
    folder / "model_evaluation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

sns.set_theme(style="white")

for name, cm in confusion_matrices.items():
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"]
    )

    plt.title(f"Матриця плутанини — {name}")
    plt.xlabel("Передбачений клас")
    plt.ylabel("Фактичний клас")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        folder / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

min_fn = results_df["FN"].min()

best_models = results_df.loc[
    results_df["FN"] == min_fn,
    "Модель"
].tolist()

print("\nМОДЕЛЬ ІЗ НАЙМЕНШОЮ КІЛЬКІСТЮ ПРОПУЩЕНИХ АТАК")

for model_name in best_models:
    print(model_name)

print(f"Кількість пропущених атак (FN): {min_fn}")

print("\nПОРІВНЯННЯ RECALL І FN")

display(
    results_df[
        ["Модель", "Recall", "FN"]
    ].reset_index(drop=True)
)

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Модель"],
    results_df["FN"]
)

plt.title("Кількість пропущених атак різними моделями")
plt.xlabel("Модель")
plt.ylabel("Кількість пропущених атак (FN)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    folder / "models_false_negatives_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\nФайли збережено:")
print("model_evaluation_results.csv")
print("Матриці плутанини для кожної моделі")
print("models_false_negatives_comparison.png")