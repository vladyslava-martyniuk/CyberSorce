import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    r2_score
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

column_aliases = {
    "destinationport": "Destination_Port",
    "destination_port": "Destination_Port",
    "flowduration": "Flow_Duration",
    "flow_duration": "Flow_Duration",
    "flowbytespers": "Flow_Bytes_per_s",
    "flow_bytes_per_s": "Flow_Bytes_per_s",
    "totalfw dpackets": "Total_Fwd_Packets",
    "totalfwdpackets": "Total_Fwd_Packets",
    "total_fwd_packets": "Total_Fwd_Packets",
    "totalbackwardpackets": "Total_Backward_Packets",
    "total_backward_packets": "Total_Backward_Packets",
    "totalbytessent": "Total Bytes Sent",
    "total_bytes_sent": "Total Bytes Sent",
    "totalbytesreceived": "Total Bytes Received",
    "total_bytes_received": "Total Bytes Received",
    "duration": "duration",
    "label": "Label"
}

rename_columns = {}

for column in df.columns:
    normalized = column.strip().lower().replace(" ", "").replace("-", "")
    normalized_underscore = column.strip().lower().replace(" ", "_")

    if normalized in column_aliases:
        rename_columns[column] = column_aliases[normalized]
    elif normalized_underscore in column_aliases:
        rename_columns[column] = column_aliases[normalized_underscore]

df = df.rename(columns=rename_columns)

target_column = next(
    (column for column in df.columns if column.lower() == "label"),
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
        "У Label мають бути лише BENIGN і ATTACK. "
        "Перевір назви класів у датасеті."
    )

df[target_column] = df[target_column].astype(int)

if df[target_column].nunique() < 2:
    raise ValueError(
        "Для класифікації потрібні обидва класи: BENIGN та ATTACK."
    )

df_cleaned = df.copy()

if {
    "Total Bytes Sent",
    "Total Bytes Received"
}.issubset(df_cleaned.columns):
    df_cleaned["total_bytes"] = (
        df_cleaned["Total Bytes Sent"]
        + df_cleaned["Total Bytes Received"]
    )

if {
    "Total_Fwd_Packets",
    "Total_Backward_Packets"
}.issubset(df_cleaned.columns):
    df_cleaned["total_packets"] = (
        df_cleaned["Total_Fwd_Packets"]
        + df_cleaned["Total_Backward_Packets"]
    )

if {
    "Total_Fwd_Packets",
    "Total_Backward_Packets"
}.issubset(df_cleaned.columns):
    df_cleaned["fwd_bwd_packet_ratio"] = (
        df_cleaned["Total_Fwd_Packets"]
        / df_cleaned["Total_Backward_Packets"].replace(0, np.nan)
    )

if {
    "total_bytes",
    "total_packets"
}.issubset(df_cleaned.columns):
    df_cleaned["bytes_per_packet_mean"] = (
        df_cleaned["total_bytes"]
        / df_cleaned["total_packets"].replace(0, np.nan)
    )

if {
    "total_packets",
    "Flow_Duration"
}.issubset(df_cleaned.columns):
    df_cleaned["packet_growth_rate"] = (
        df_cleaned["total_packets"]
        / df_cleaned["Flow_Duration"].replace(0, np.nan)
    )

if {
    "total_bytes",
    "Flow_Duration"
}.issubset(df_cleaned.columns):
    df_cleaned["Flow_Bytes_per_s"] = (
        df_cleaned["total_bytes"]
        / df_cleaned["Flow_Duration"].replace(0, np.nan)
    )

df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)

df_transformed = df_cleaned.copy()

print("РОЗМІР ДАТАСЕТУ")
print(f"Кількість рядків: {df_transformed.shape[0]}")
print(f"Кількість стовпців: {df_transformed.shape[1]}")
print("\nРОЗПОДІЛ КЛАСІВ")
print(df_transformed["Label"].value_counts().rename({
    0: "BENIGN",
    1: "ATTACK"
}))

# -------------------------------------------------------------
# ЗАВДАННЯ 1. КЛАСИФІКАЦІЯ МЕРЕЖЕВОГО ТРАФІКУ
# -------------------------------------------------------------

X = df_transformed.drop(columns=["Label"])
y = df_transformed["Label"]

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

classification_models = {
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

classification_results = []
confusion_matrices = {}

for name, model in classification_models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()
    confusion_matrices[name] = cm

    classification_results.append({
        "Модель": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

classification_df = pd.DataFrame(classification_results)

classification_df = classification_df.sort_values(
    by=["FN", "Recall", "F1-Score"],
    ascending=[True, False, False]
)

print("\nРЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ")
print(classification_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

classification_df.to_csv(
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
    plt.close()

min_fn = classification_df["FN"].min()

best_models = classification_df.loc[
    classification_df["FN"] == min_fn,
    "Модель"
].tolist()

print("\nМОДЕЛІ З НАЙМЕНШОЮ КІЛЬКІСТЮ ПРОПУЩЕНИХ АТАК")

for model_name in best_models:
    print(model_name)

print(f"Мінімальна кількість FN: {min_fn}")

print("\nПОРІВНЯННЯ RECALL І FN")
print(
    classification_df[
        ["Модель", "Recall", "FN"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

plt.figure(figsize=(10, 5))
plt.bar(
    classification_df["Модель"],
    classification_df["FN"]
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
plt.close()

# -------------------------------------------------------------
# ЗАВДАННЯ 2. ЕКСТРЕМАЛЬНІ ЗНАЧЕННЯ ДЛЯ ПОРТІВ
# -------------------------------------------------------------

print("\nЕКСТРЕМАЛЬНІ ЗНАЧЕННЯ ДЛЯ ПОРТІВ 443, 80, 53")

top_ports = [443, 80, 53]

required_port_columns = [
    "Destination_Port",
    "Flow_Duration",
    "Flow_Bytes_per_s"
]

if all(column in df_cleaned.columns for column in required_port_columns):
    for port in top_ports:
        port_data = df_cleaned[
            df_cleaned["Destination_Port"] == port
        ]

        if not port_data.empty:
            max_duration = port_data["Flow_Duration"].max()
            min_duration = port_data["Flow_Duration"].min()

            max_bytes = port_data["Flow_Bytes_per_s"].max()
            min_bytes = port_data["Flow_Bytes_per_s"].min()

            print(f"\nПорт: {port}")
            print(
                f"Flow Duration — максимум: {max_duration}, "
                f"мінімум: {min_duration}"
            )
            print(
                f"Flow Bytes/s — максимум: {max_bytes}, "
                f"мінімум: {min_bytes}"
            )
        else:
            print(f"\nДля порту {port} даних немає.")
else:
    missing_columns = [
        column for column in required_port_columns
        if column not in df_cleaned.columns
    ]

    print(
        "Завдання пропущено: у датасеті немає потрібних стовпців: "
        + ", ".join(missing_columns)
    )

# -------------------------------------------------------------
# ЗАВДАННЯ 3. РЕГРЕСІЯ ДЛЯ FLOW_BYTES_PER_S
# -------------------------------------------------------------

print("\nРЕГРЕСІЙНИЙ АНАЛІЗ")

target_regression = "Flow_Bytes_per_s"

if target_regression not in df_transformed.columns:
    print(
        "Регресію неможливо виконати: "
        "у датасеті немає цільового стовпця Flow_Bytes_per_s."
    )
else:
    features_cols = [
        "Flow_Duration",
        "Total_Fwd_Packets",
        "Total_Backward_Packets",
        "fwd_bwd_packet_ratio",
        "bytes_per_packet_mean",
        "packet_growth_rate"
    ]

    available_features = [
        column for column in features_cols
        if column in df_transformed.columns
    ]

    if not available_features:
        print(
            "Регресію неможливо виконати: "
            "не знайдено потрібних ознак."
        )
    else:
        regression_df = df_transformed[
            available_features + [target_regression]
        ].copy()

        regression_df = regression_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        regression_df = regression_df.dropna(
            subset=[target_regression]
        )

        X_reg = regression_df[available_features]
        y_reg = regression_df[target_regression]

        valid_rows = y_reg.notna()
        X_reg = X_reg.loc[valid_rows]
        y_reg = y_reg.loc[valid_rows]

        if len(regression_df) < 5:
            print(
                "Недостатньо рядків для надійного виконання регресії."
            )
        else:
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_reg,
                y_reg,
                test_size=0.2,
                random_state=42
            )

            regression_preprocessor = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            X_train_reg_scaled = regression_preprocessor.fit_transform(
                X_train_reg
            )

            X_test_reg_scaled = regression_preprocessor.transform(
                X_test_reg
            )

            regression_models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=1.0, max_iter=10000)
            }

            regression_results = {}

            for name, model in regression_models.items():
                model.fit(
                    X_train_reg_scaled,
                    y_train_reg
                )

                y_pred_reg = model.predict(
                    X_test_reg_scaled
                )

                regression_results[name] = {
                    "MSE": mean_squared_error(
                        y_test_reg,
                        y_pred_reg
                    ),
                    "R2": r2_score(
                        y_test_reg,
                        y_pred_reg
                    )
                }

            poly = PolynomialFeatures(
                degree=2,
                include_bias=False
            )

            X_train_poly = poly.fit_transform(
                X_train_reg_scaled
            )

            X_test_poly = poly.transform(
                X_test_reg_scaled
            )

            poly_lr = LinearRegression()

            poly_lr.fit(
                X_train_poly,
                y_train_reg
            )

            y_pred_poly = poly_lr.predict(
                X_test_poly
            )

            regression_results[
                "Polynomial Regression (degree 2)"
            ] = {
                "MSE": mean_squared_error(
                    y_test_reg,
                    y_pred_poly
                ),
                "R2": r2_score(
                    y_test_reg,
                    y_pred_poly
                )
            }

            regression_df_results = pd.DataFrame(
                regression_results
            ).T

            regression_df_results = regression_df_results.sort_values(
                by="MSE"
            )

            print("\nРЕЗУЛЬТАТИ РЕГРЕСІЇ")
            print(
                regression_df_results.to_string(
                    float_format=lambda x: f"{x:.6f}"
                )
            )

            regression_df_results.to_csv(
                folder / "regression_evaluation_results.csv",
                encoding="utf-8-sig"
            )

            best_model_name = regression_df_results["MSE"].idxmin()

            print(
                f"\nНайкраща модель за MSE: {best_model_name}"
            )

            print(
                f"MSE: {regression_df_results.loc[best_model_name, 'MSE']:.6f}"
            )

            print(
                f"R²: {regression_df_results.loc[best_model_name, 'R2']:.6f}"
            )

            plt.figure(figsize=(10, 5))

            plt.bar(
                regression_df_results.index,
                regression_df_results["MSE"]
            )

            plt.title("Порівняння регресійних моделей за MSE")
            plt.xlabel("Модель")
            plt.ylabel("MSE")
            plt.xticks(rotation=15)
            plt.tight_layout()

            plt.savefig(
                folder / "regression_models_mse_comparison.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.show()
            plt.close()

print("\nРоботу завершено.")
print("Результати класифікації: model_evaluation_results.csv")
print("Результати регресії: regression_evaluation_results.csv")