import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
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
    "totalfwdpackets": "Total_Fwd_Packets",
    "total_fwd_packets": "Total_Fwd_Packets",
    "totalbackwardpackets": "Total_Backward_Packets",
    "total_backward_packets": "Total_Backward_Packets",
    "totalbytessent": "Total Bytes Sent",
    "total_bytes_sent": "Total Bytes Sent",
    "totalbytesreceived": "Total Bytes Received",
    "total_bytes_received": "Total Bytes Received",
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
    .replace({
        "BENIGN": "0",
        "NORMAL": "0",
        "ATTACK": "1",
        "MALICIOUS": "1"
    })
)

if not set(df[target_column].unique()).issubset({"0", "1"}):
    raise ValueError(
        "У Label мають бути класи BENIGN/ATTACK або 0/1."
    )

df[target_column] = df[target_column].astype(int)

if df[target_column].nunique() < 2:
    raise ValueError(
        "Для класифікації потрібні обидва класи: BENIGN та ATTACK."
    )

df_cleaned = df.copy()

if {"Total Bytes Sent", "Total Bytes Received"}.issubset(df_cleaned.columns):
    df_cleaned["total_bytes"] = (
        pd.to_numeric(df_cleaned["Total Bytes Sent"], errors="coerce")
        + pd.to_numeric(df_cleaned["Total Bytes Received"], errors="coerce")
    )

if {"Total_Fwd_Packets", "Total_Backward_Packets"}.issubset(df_cleaned.columns):
    df_cleaned["total_packets"] = (
        pd.to_numeric(df_cleaned["Total_Fwd_Packets"], errors="coerce")
        + pd.to_numeric(df_cleaned["Total_Backward_Packets"], errors="coerce")
    )

if {"Total_Fwd_Packets", "Total_Backward_Packets"}.issubset(df_cleaned.columns):
    df_cleaned["fwd_bwd_packet_ratio"] = (
        pd.to_numeric(df_cleaned["Total_Fwd_Packets"], errors="coerce")
        / pd.to_numeric(
            df_cleaned["Total_Backward_Packets"], errors="coerce"
        ).replace(0, np.nan)
    )

if {"total_bytes", "total_packets"}.issubset(df_cleaned.columns):
    df_cleaned["bytes_per_packet_mean"] = (
        df_cleaned["total_bytes"]
        / df_cleaned["total_packets"].replace(0, np.nan)
    )

if {"total_packets", "Flow_Duration"}.issubset(df_cleaned.columns):
    df_cleaned["packet_growth_rate"] = (
        df_cleaned["total_packets"]
        / pd.to_numeric(
            df_cleaned["Flow_Duration"], errors="coerce"
        ).replace(0, np.nan)
    )

if {"total_bytes", "Flow_Duration"}.issubset(df_cleaned.columns):
    df_cleaned["Flow_Bytes_per_s"] = (
        df_cleaned["total_bytes"]
        / pd.to_numeric(
            df_cleaned["Flow_Duration"], errors="coerce"
        ).replace(0, np.nan)
    )

df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)
df_transformed = df_cleaned.copy()

print("РОЗМІР ДАТАСЕТУ")
print(f"Кількість рядків: {df_transformed.shape[0]}")
print(f"Кількість стовпців: {df_transformed.shape[1]}")

print("\nРОЗПОДІЛ КЛАСІВ")
print(
    df_transformed["Label"].value_counts().rename({
        0: "BENIGN",
        1: "ATTACK"
    })
)

results_folder = folder / "results_cybersecurity"
results_folder.mkdir(parents=True, exist_ok=True)

df_transformed.to_csv(
    results_folder / "cybersecurity_network_traffic_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

sns.set_theme(style="white")

X = df_transformed.drop(columns=["Label"]).copy()
y = df_transformed["Label"].copy()

X = X.drop(
    columns=[
        column for column in X.columns
        if column.lower() in {"label", "target", "class"}
    ],
    errors="ignore"
)

X = X.replace([np.inf, -np.inf], np.nan)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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

preprocessor = ColumnTransformer(transformers=transformers)

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
        random_state=42,
        n_jobs=-1
    ),
    "k-NN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

classification_results = []
confusion_matrices = {}
fitted_classification_models = {}

for name, model in classification_models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    confusion_matrices[name] = cm
    fitted_classification_models[name] = pipeline

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
print(
    classification_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

classification_df.to_csv(
    results_folder / "model_evaluation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

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
        results_folder / filename,
        dpi=300,
        bbox_inches="tight"
    )
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
    results_folder / "models_false_negatives_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

rf_f1 = classification_df.loc[
    classification_df["Модель"] == "Random Forest",
    "F1-Score"
].iloc[0]

gb_f1 = classification_df.loc[
    classification_df["Модель"] == "Gradient Boosting",
    "F1-Score"
].iloc[0]

print(
    f"\nRandom Forest F1: {rf_f1:.4f} | "
    f"Gradient Boosting F1: {gb_f1:.4f}"
)

print("\nОПТИМІЗАЦІЯ RANDOM FOREST")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_rf_optimized = best_rf.predict(X_test)

cm_optimized = confusion_matrix(
    y_test,
    y_pred_rf_optimized,
    labels=[0, 1]
)

tn, fp, fn, tp = cm_optimized.ravel()

optimized_rf_results = {
    "Модель": "Random Forest optimized",
    "Accuracy": accuracy_score(y_test, y_pred_rf_optimized),
    "Precision": precision_score(
        y_test, y_pred_rf_optimized, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred_rf_optimized, zero_division=0
    ),
    "F1-Score": f1_score(
        y_test, y_pred_rf_optimized, zero_division=0
    ),
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}

rf_original_results = classification_df[
    classification_df["Модель"] == "Random Forest"
].iloc[0].to_dict()

rf_comparison_df = pd.DataFrame([
    rf_original_results,
    optimized_rf_results
])

rf_comparison_df.to_csv(
    results_folder / "random_forest_before_after.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(grid_search.cv_results_).to_csv(
    results_folder / "random_forest_grid_search_results.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    results_folder / "random_forest_best_params.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Найкращі параметри: {grid_search.best_params_}\n")
    file.write(f"Найкращий F1 під час CV: {grid_search.best_score_:.6f}\n")
    file.write(
        f"F1 до оптимізації: "
        f"{rf_original_results['F1-Score']:.6f}\n"
    )
    file.write(
        f"F1 після оптимізації: "
        f"{optimized_rf_results['F1-Score']:.6f}\n"
    )

print("Параметри до оптимізації:")
print({
    "n_estimators": 100,
    "random_state": 42
})
print("Найкращі параметри:", grid_search.best_params_)
print(f"F1 до оптимізації: {rf_original_results['F1-Score']:.4f}")
print(f"F1 після оптимізації: {optimized_rf_results['F1-Score']:.4f}")

plt.figure(figsize=(8, 5))
plt.bar(
    rf_comparison_df["Модель"],
    rf_comparison_df["F1-Score"]
)
plt.title("Random Forest до та після оптимізації")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(
    results_folder / "random_forest_optimization_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nЕКСТРЕМАЛЬНІ ЗНАЧЕННЯ ДЛЯ ПОРТІВ 443, 80, 53")

top_ports = [443, 80, 53]

required_port_columns = [
    "Destination_Port",
    "Flow_Duration",
    "Flow_Bytes_per_s"
]

if all(column in df_cleaned.columns for column in required_port_columns):
    for column in required_port_columns:
        df_cleaned[column] = pd.to_numeric(
            df_cleaned[column], errors="coerce"
        )

    for port in top_ports:
        port_data = df_cleaned[
            df_cleaned["Destination_Port"] == port
        ]

        if not port_data.empty:
            print(f"\nПорт: {port}")
            print(
                f"Flow Duration — максимум: "
                f"{port_data['Flow_Duration'].max()}, "
                f"мінімум: {port_data['Flow_Duration'].min()}"
            )
            print(
                f"Flow Bytes/s — максимум: "
                f"{port_data['Flow_Bytes_per_s'].max()}, "
                f"мінімум: {port_data['Flow_Bytes_per_s'].min()}"
            )
        else:
            print(f"\nДля порту {port} даних немає.")
else:
    missing_columns = [
        column for column in required_port_columns
        if column not in df_cleaned.columns
    ]
    print(
        "Завдання пропущено: немає потрібних стовпців: "
        + ", ".join(missing_columns)
    )

print("\nРЕГРЕСІЙНИЙ АНАЛІЗ")

target_regression = "Flow_Bytes_per_s"

if target_regression not in df_transformed.columns:
    print(
        "Регресію неможливо виконати: "
        "немає стовпця Flow_Bytes_per_s."
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
        print("Не знайдено ознак для регресії.")
    else:
        regression_df = df_transformed[
            available_features + [target_regression]
        ].copy()

        for column in regression_df.columns:
            regression_df[column] = pd.to_numeric(
                regression_df[column], errors="coerce"
            )

        regression_df = regression_df.replace(
            [np.inf, -np.inf], np.nan
        )

        regression_df = regression_df.dropna(
            subset=[target_regression]
        )

        regression_df = regression_df[
            regression_df[target_regression].notna()
        ]

        if len(regression_df) < 5:
            print("Недостатньо рядків для регресії.")
        else:
            X_reg = regression_df[available_features]
            y_reg = regression_df[target_regression]

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
                "Lasso Regression": Lasso(
                    alpha=1.0,
                    max_iter=10000
                )
            }

            regression_results = {}
            fitted_reg_models = {}
            regression_predictions = {}

            for name, model in regression_models.items():
                model.fit(X_train_reg_scaled, y_train_reg)
                y_pred_reg = model.predict(X_test_reg_scaled)

                fitted_reg_models[name] = model
                regression_predictions[name] = y_pred_reg

                regression_results[name] = {
                    "MAE": mean_absolute_error(y_test_reg, y_pred_reg),
                    "MSE": mean_squared_error(y_test_reg, y_pred_reg),
                    "RMSE": np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
                    "R2": r2_score(y_test_reg, y_pred_reg)
                }

            poly = PolynomialFeatures(
                degree=2,
                include_bias=False
            )

            X_train_poly = poly.fit_transform(X_train_reg_scaled)
            X_test_poly = poly.transform(X_test_reg_scaled)

            poly_lr = LinearRegression()
            poly_lr.fit(X_train_poly, y_train_reg)

            y_pred_poly = poly_lr.predict(X_test_poly)

            fitted_reg_models["Polynomial Regression (degree 2)"] = (
                poly_lr,
                poly
            )
            regression_predictions["Polynomial Regression (degree 2)"] = (
                y_pred_poly
            )

            regression_results["Polynomial Regression (degree 2)"] = {
                "MAE": mean_absolute_error(y_test_reg, y_pred_poly),
                "MSE": mean_squared_error(y_test_reg, y_pred_poly),
                "RMSE": np.sqrt(mean_squared_error(y_test_reg, y_pred_poly)),
                "R2": r2_score(y_test_reg, y_pred_poly)
            }

            regression_df_results = pd.DataFrame(
                regression_results
            ).T.sort_values(by="MSE")

            print("\nРЕЗУЛЬТАТИ РЕГРЕСІЇ")
            print(
                regression_df_results.to_string(
                    float_format=lambda x: f"{x:.6f}"
                )
            )

            regression_df_results.to_csv(
                results_folder / "regression_evaluation_results.csv",
                encoding="utf-8-sig"
            )

            best_model_name = regression_df_results["MSE"].idxmin()
            best_y_pred = regression_predictions[best_model_name]

            print(f"\nНайкраща модель за MSE: {best_model_name}")
            print(
                f"MAE: "
                f"{regression_df_results.loc[best_model_name, 'MAE']:.6f}"
            )
            print(
                f"MSE: "
                f"{regression_df_results.loc[best_model_name, 'MSE']:.6f}"
            )
            print(
                f"RMSE: "
                f"{regression_df_results.loc[best_model_name, 'RMSE']:.6f}"
            )
            print(
                f"R²: "
                f"{regression_df_results.loc[best_model_name, 'R2']:.6f}"
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
                results_folder / "regression_models_mse_comparison.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.scatter(
                y_test_reg,
                best_y_pred,
                alpha=0.4,
                color="teal"
            )
            min_value = min(y_test_reg.min(), np.min(best_y_pred))
            max_value = max(y_test_reg.max(), np.max(best_y_pred))
            plt.plot(
                [min_value, max_value],
                [min_value, max_value],
                "r--",
                linewidth=2
            )
            plt.title(f"Predicted vs Actual ({best_model_name})")
            plt.xlabel("Реальні значення (Actual)")
            plt.ylabel("Прогнозовані значення (Predicted)")
            plt.tight_layout()
            plt.savefig(
                results_folder / "regression_pred_vs_actual.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

            residuals = y_test_reg.to_numpy() - best_y_pred

            plt.figure(figsize=(8, 4))
            sns.histplot(
                residuals,
                kde=True,
                color="crimson",
                bins=30
            )
            plt.title(
                f"Розподіл помилок (Residuals) — {best_model_name}"
            )
            plt.xlabel("Помилка (Actual - Predicted)")
            plt.ylabel("Кількість")
            plt.tight_layout()
            plt.savefig(
                results_folder / "regression_residuals.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

            errors_df = pd.DataFrame({
                "Actual": y_test_reg.to_numpy(),
                "Predicted": best_y_pred,
                "Absolute_Error": np.abs(
                    y_test_reg.to_numpy() - best_y_pred
                ),
                "Signed_Error": y_test_reg.to_numpy() - best_y_pred
            })

            errors_df["Relative_Error_Percent"] = np.where(
                np.abs(errors_df["Actual"]) > 1e-12,
                errors_df["Absolute_Error"]
                / np.abs(errors_df["Actual"]) * 100,
                np.nan
            )

            errors_df = errors_df.sort_values(
                by="Absolute_Error",
                ascending=False
            )

            top_10_errors = errors_df.head(10)

            top_10_errors.to_csv(
                results_folder / "top_10_regression_errors.csv",
                index=False,
                encoding="utf-8-sig"
            )

            errors_df.to_csv(
                results_folder / "all_regression_errors.csv",
                index=False,
                encoding="utf-8-sig"
            )

            print("\n10 НАЙБІЛЬШИХ ПОМИЛОК РЕГРЕСІЇ")
            print(
                top_10_errors.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}"
                )
            )

            print("\nСТАТИСТИКА ПОМИЛОК")
            print(errors_df["Absolute_Error"].describe())

            plt.figure(figsize=(10, 5))
            plt.bar(
                range(1, len(top_10_errors) + 1),
                top_10_errors["Absolute_Error"]
            )
            plt.title("10 найбільших абсолютних помилок регресії")
            plt.xlabel("Прогноз")
            plt.ylabel("Абсолютна помилка")
            plt.tight_layout()
            plt.savefig(
                results_folder / "top_10_regression_errors.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

print("\nУСІ ЗАВДАННЯ ЗАВЕРШЕНО")
print(f"Результати збережено в папці: {results_folder}")