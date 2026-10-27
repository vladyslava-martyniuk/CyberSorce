import pandas as pd
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

df = pd.read_csv("cybersecurity_network_traffic_synthetic.csv")


def check_data_quality(df):
    """
    Перевіряє наявність пропущених значень і дублікатів.
    """
    if df.empty:
        print("DataFrame порожній. Перевірка неможлива.")
        return

    print("Кількість пропущених значень у кожному стовпці:")
    print(df.isnull().sum())

    print(f"\nЗагальна кількість пропущених значень: {df.isnull().sum().sum()}")
    print(f"Кількість дублікатів: {df.duplicated().sum()}")


def clean_data_quality(df):
    """
    Заповнює пропущені значення медіаною для числових стовпців
    і найчастішим значенням для текстових.
    Видаляє дублікати.
    """
    if df.empty:
        print("DataFrame порожній. Очищення неможливе.")
        return df.copy()

    df_cleaned = df.copy()

    rows_before = len(df_cleaned)

   
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()

        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                
                median_value = df_cleaned[column].median()

                if pd.notna(median_value):
                    df_cleaned[column] = df_cleaned[column].fillna(
                        median_value
                    )
                else:
                    print(
                        f"Увага: стовпець {column} не має "
                        "доступних числових значень для обчислення медіани."
                    )

            else:
                
                mode_values = df_cleaned[column].mode()

                if not mode_values.empty:
                    df_cleaned[column] = df_cleaned[column].fillna(
                        mode_values[0]
                    )
                else:
                    print(
                        f"Увага: стовпець {column} не має "
                        "значень для заповнення пропусків."
                    )

    df_cleaned = df_cleaned.drop_duplicates()

    rows_after = len(df_cleaned)

    print(f"\nКількість рядків до очищення: {rows_before}")
    print(f"Кількість рядків після очищення: {rows_after}")
    print(f"Видалено дублікатів: {rows_before - rows_after}")

    print(
        "\nКількість пропущених значень після очищення:",
        df_cleaned.isnull().sum().sum()
    )

    return df_cleaned



print("=== ПЕРЕВІРКА ЯКОСТІ ДАНИХ ===")
check_data_quality(df)


print("\n=== ОЧИЩЕННЯ ДАНИХ ===")
df_cleaned = clean_data_quality(df)


df_cleaned.to_csv(
    "cybersecurity_network_traffic_cleaned.csv",
    index=False
)

print("\nОчищений датасет збережено.")

sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df_cleaned,
    x="Flow_Duration",
    hue="Label",
    multiple="dodge",
    bins=20
)

plt.title(
    "Розподіл тривалості мережевих потоків за типами трафіку"
)
plt.xlabel("Тривалість потоку (Flow Duration)")
plt.ylabel("Кількість")

plt.savefig("traffic_visualization.png")
print("Графік успішно збережено у файл traffic_visualization та готовий до звіту")
plt.show()


df_transformed = pd.get_dummies(
    df, 
    columns=['Protocol', 'Connection_State'], 
    prefix=['Protocol', 'State'], 
    dtype=int
)


df_transformed['fwd_bwd_packet_ratio'] = df_transformed['Total_Fwd_Packets'] / (df_transformed['Total_Backward_Packets'] + 1e-5)
df_transformed['bytes_per_packet_mean'] = df_transformed['Flow_Bytes_per_s'] / (df_transformed['Flow_Packets_per_s'] + 1e-5)
df_transformed['packet_growth_rate'] = (df_transformed['Total_Fwd_Packets'] - df_transformed['Total_Backward_Packets']) / df_transformed['Flow_Duration']

output_filename = 'cybersecurity_network_traffic_transformed.csv'
df_transformed.to_csv(output_filename, index=False)

print(f"Трансформований датасет збережено у файл: {output_filename}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Flow_Duration'], kde=True, color='blue')
plt.title('Розподіл тривалості потоку (Flow_Duration)')

plt.subplot(1, 2, 2)
sns.histplot(df['Flow_Bytes_per_s'], kde=True, color='green')
plt.title('Розподіл інтенсивності трафіку (Flow_Bytes_per_s)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Protocol', y='Flow_Bytes_per_s', data=df)
plt.title('Порівняння інтенсивності трафіку (Flow_Bytes_per_s) за протоколами')
plt.show()

pairplot_cols = ['Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Flow_Bytes_per_s']
sns.pairplot(df[pairplot_cols], diag_kind='kde')
plt.show()


# -------------------------------------------------------------
# ЗАВДАННЯ 5: Лінійні графіки трендів
# -------------------------------------------------------------
top_ports = [443, 80, 53]
df_selected = df_cleaned[df_cleaned['Destination_Port'].isin(top_ports)].copy()
df_selected['Stream_ID'] = df_selected.groupby('Destination_Port').cumcount()

plt.figure(figsize=(14, 6))

# Тренд Flow_Duration
plt.subplot(1, 2, 1)
sns.lineplot(data=df_selected, x='Stream_ID', y='Flow_Duration', hue='Destination_Port', palette='Set2')
plt.title("Тренд Flow Duration для портів (443, 80, 53)")
plt.xlabel("Порядковий номер потоку")
plt.ylabel("Flow Duration")



# Тренд Flow_Bytes_per_s
plt.subplot(1, 2, 2)
sns.lineplot(data=df_selected, x='Stream_ID', y='Flow_Bytes_per_s', hue='Destination_Port', palette='Set2')
plt.title("Тренд Flow Bytes/s для портів (443, 80, 53)")
plt.xlabel("Порядковий номер потоку")
plt.ylabel("Flow Bytes per second")

plt.tight_layout()
plt.savefig("trends_ports_analysis.png")
print("Графік трендів збережено у файл trends_ports_analysis.png")
plt.show()

# -------------------------------------------------------------
# ЗАВДАННЯ 5: Бар-чарт топ-10
# -------------------------------------------------------------
top10_ports = df_cleaned.groupby('Destination_Port')['Flow_Bytes_per_s'].mean().nlargest(10).reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=top10_ports, x='Destination_Port', y='Flow_Bytes_per_s', palette='viridis', order=top10_ports['Destination_Port'])
plt.title("Топ-10 Destination_Port за середнім Flow_Bytes_per_s")
plt.xlabel("Destination Port")
plt.ylabel("Середній Flow Bytes/s")

plt.tight_layout()
plt.savefig("top10_ports_traffic.png")
print("Бар-чарт топ-10 портів збережено у файл top10_ports_traffic.png")
plt.show()

# -------------------------------------------------------------
# ЗАВДАННЯ 5 Найвищі та найнижчі значення 
# -------------------------------------------------------------
print("\n=== Екстремальні значення метрик для портів 443, 80, 53 ===")
for port in top_ports:
    port_data = df_cleaned[df_cleaned['Destination_Port'] == port]
    if not port_data.empty:
        max_duration = port_data['Flow_Duration'].max()
        min_duration = port_data['Flow_Duration'].min()
        
        max_bytes = port_data['Flow_Bytes_per_s'].max()
        min_bytes = port_data['Flow_Bytes_per_s'].min()
        
        print(f"\nПорт: {port}")
        print(f"  [Flow Duration]   Максимум: {max_duration} | Мінімум: {min_duration}")
        print(f"  [Flow Bytes/s]    Максимум: {max_bytes} | Мінімум: {min_bytes}")


features_cols = [
    'Flow_Duration',
    'Total_Fwd_Packets',
    'Total_Backward_Packets',
    'fwd_bwd_packet_ratio',
    'bytes_per_packet_mean',
    'packet_growth_rate',
]

X = df_transformed.drop(columns=['Flow_Bytes_per_s', 'Label'], errors='ignore')
X = X.select_dtypes(include=['number'])

y = df_transformed['Flow_Bytes_per_s']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

# -------------------------------------------------------------
#                       Linear Regression
# -------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results["Linear Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_lr),
    "R2": r2_score(y_test, y_pred_lr),
}

# -------------------------------------------------------------
#              Polynomial Regression (ступінь 2)
# -------------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_pred_poly = poly_lr.predict(X_test_poly)
results["Polynomial Regression (deg=2)"] = {
    "MSE": mean_squared_error(y_test, y_pred_poly),
    "R2": r2_score(y_test, y_pred_poly),
}

# -------------------------------------------------------------
#                       Ridge Regression
# -------------------------------------------------------------
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
results["Ridge Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_ridge),
    "R2": r2_score(y_test, y_pred_ridge),
}

# -------------------------------------------------------------
#                       Lasso Regression
# -------------------------------------------------------------
lasso = Lasso(alpha=1.0, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
results["Lasso Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_lasso),
    "R2": r2_score(y_test, y_pred_lasso),
}


results_df = pd.DataFrame(results).T
print(results_df)


best_model_name = results_df["MSE"].idxmin()
print(
    f"\nНайкраща модель за показником MSE: **{best_model_name}** з мін. помилкою."
)


regression_results = {}


# -------------------------------------------------------------
#                       Linear Regression
# -------------------------------------------------------------
y_pred_lr = lr.predict(X_test)
regression_results["Linear Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_lr),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    "MAE": mean_absolute_error(y_test, y_pred_lr),
    "R2": r2_score(y_test, y_pred_lr),
}

# -------------------------------------------------------------
#                       Polynomial Regression (deg=2)
# -------------------------------------------------------------
y_pred_poly = poly_lr.predict(X_test_poly)
regression_results["Polynomial Regression (deg=2)"] = {
    "MSE": mean_squared_error(y_test, y_pred_poly),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_poly)),
    "MAE": mean_absolute_error(y_test, y_pred_poly),
    "R2": r2_score(y_test, y_pred_poly),
}

# -------------------------------------------------------------
#                       Ridge Regression
# -------------------------------------------------------------
y_pred_ridge = ridge.predict(X_test)
regression_results["Ridge Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_ridge),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    "MAE": mean_absolute_error(y_test, y_pred_ridge),
    "R2": r2_score(y_test, y_pred_ridge),
}

# -------------------------------------------------------------
#                       Lasso Regression
# -------------------------------------------------------------
y_pred_lasso = lasso.predict(X_test)
regression_results["Lasso Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_lasso),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    "MAE": mean_absolute_error(y_test, y_pred_lasso),
    "R2": r2_score(y_test, y_pred_lasso),
}


reg_results_df = pd.DataFrame(regression_results).T
print("\n=== ПОРІВНЯННЯ РЕГРЕСІЙНИХ МОДЕЛЕЙ ===")
print(reg_results_df)


best_reg_model = reg_results_df["RMSE"].idxmin()
print(
    f"\nНайкраща модель для прогнозування: **{best_reg_model}** (найменша похибка RMSE)."
)


X_clf = df_transformed.drop(columns=['Flow_Bytes_per_s', 'Label'], errors='ignore')
X_clf = X_clf.select_dtypes(include=['number'])


if 'Label' in df_transformed.columns:
    y_clf = df_transformed['Label']
elif 'Label' in df.columns:
    y_clf = df['Label']
else:
    raise ValueError("Цільова змінна 'Label' не знайдена у датасеті.")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


classification_models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
}

cv_results = {}

print("\n=== ПРОЦЕС КРОС-ВАЛІДАЦІЇ (5-Fold, F1-Score) ===")

for model_name, model in classification_models.items():

    scores = cross_val_score(model, X_clf, y_clf, cv=skf, scoring='f1_macro')
    
    mean_f1 = scores.mean()
    std_f1 = scores.std()
    
    cv_results[model_name] = {
        "Mean F1-Score": mean_f1,
        "Std (Standard Deviation)": std_f1,
        "Scores": scores
    }
    
    print(f"Модель: {model_name}")
    print(f"  > Середній F1-Score: {mean_f1}")
    print(f"  > Стандартне відхилення (Std): {std_f1}\n")

cv_summary_df = pd.DataFrame({
    "Mean F1-Score": [res["Mean F1-Score"] for res in cv_results.values()],
    "Std": [res["Std (Standard Deviation)"] for res in cv_results.values()]
}, index=cv_results.keys())

print("\n=== ПІДСУМОК КРОС-ВАЛІДАЦІЇ КЛАСИФІКАТОРІВ ===")
print(cv_summary_df)


most_stable_model = cv_summary_df["Std"].idxmin()
print(
    f"\nНайстабільніша модель (найменше стандартне відхилення): {most_stable_model} "
    f"з показником Std = {cv_summary_df.loc[most_stable_model, 'Std']}"
)