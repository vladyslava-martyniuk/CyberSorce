from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "cybersecurity_network_traffic_synthetic.csv")
X = df.drop(columns="Label")
y = df["Label"].map({"BENIGN": 0, "ATTACK": 1}).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
baseline = Pipeline([("preprocess", preprocess), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
baseline.fit(X_train, y_train)
before = baseline.predict(X_test)
search = GridSearchCV(
    Pipeline([("preprocess", preprocess), ("rf", RandomForestClassifier(random_state=42))]),
    {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [5, 10, 20, None],
        "rf__min_samples_split": [2, 5, 10]
    },
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1,
    refit=True
)
search.fit(X_train, y_train)
after = search.predict(X_test)
results = pd.DataFrame([
    {"Model": "Random Forest (before)", "Accuracy": accuracy_score(y_test, before), "Precision": precision_score(y_test, before, zero_division=0), "Recall": recall_score(y_test, before, zero_division=0), "F1": f1_score(y_test, before, zero_division=0)},
    {"Model": "Random Forest (after)", "Accuracy": accuracy_score(y_test, after), "Precision": precision_score(y_test, after, zero_division=0), "Recall": recall_score(y_test, after, zero_division=0), "F1": f1_score(y_test, after, zero_division=0)}
])
results.to_csv(folder / "zavdannia_4_metrics.csv", index=False)
params = {key.split("rf__", 1)[1]: value for key, value in search.best_params_.items()}
(folder / "zavdannia_4_params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
summary = [
    "Завдання 4. Оптимізація гіперпараметрів Random Forest",
    f"Кількість записів: {len(df)}, train: {len(X_train)}, test: {len(X_test)}",
    "GridSearchCV: 5-fold, F1, 36 комбінацій",
    f"Найкращі параметри: {params}",
    f"Найкращий середній F1 у крос-валідації: {search.best_score_:.4f}",
    f"F1 на тесті до оптимізації: {f1_score(y_test, before):.4f}",
    f"F1 на тесті після оптимізації: {f1_score(y_test, after):.4f}"
]
(folder / "zavdannia_4_result.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary))
print(results.to_string(index=False))
