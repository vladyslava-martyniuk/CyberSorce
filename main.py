import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

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





folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "cybersecurity_network_traffic_synthetic.csv")
columns = [
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Backward_Packets",
    "Fwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Mean",
    "Flow_Bytes_per_s",
    "Flow_Packets_per_s",
    "Flow_IAT_Mean",
]
numeric = data[columns].apply(pd.to_numeric, errors="coerce")
stats = pd.DataFrame({
    "count": numeric.count(),
    "mean": numeric.mean(),
    "median": numeric.median(),
    "mode": numeric.mode(dropna=True).iloc[0],
    "std": numeric.std(),
    "min": numeric.min(),
    "Q1": numeric.quantile(0.25),
    "Q3": numeric.quantile(0.75),
    "max": numeric.max(),
    "skewness": numeric.skew(),
})
stats.index.name = "parameter"
stats.round(3).to_csv(folder / "statystyka.csv", encoding="utf-8-sig")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, col in zip(axes.flat, columns):
    sns.histplot(numeric[col].dropna(), bins=18, ax=ax, color="#3979a6", edgecolor="white")
    ax.axvline(numeric[col].mean(), color="#d95c3f", linewidth=1.6, label="Mean")
    ax.axvline(numeric[col].median(), color="#24684d", linewidth=1.6, linestyle="--", label="Median")
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
fig.suptitle("Network Traffic: Statistical Distributions", fontsize=17, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(folder / "statystyka_rozpodil.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(stats.round(2).to_string())

from pathlib import Path
import sys
import pandas as pd

columns = [
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Backward_Packets",
    "Fwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Mean",
    "Flow_Bytes_per_s",
    "Flow_Packets_per_s",
    "Flow_IAT_Mean",
]

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("cybersecurity_network_traffic_synthetic.csv")
df = pd.read_csv(source)
corr = df[columns].corr(method="pearson")

pairs = [
    (a, b, float(corr.loc[a, b]))
    for i, a in enumerate(columns)
    for b in columns[i + 1:]
]

positive = sorted((p for p in pairs if p[2] > 0), key=lambda p: -p[2])
negative = sorted((p for p in pairs if p[2] < 0), key=lambda p: p[2])

print(f"Записів: {len(df)}. Проаналізовано {len(columns)} кількісних ознак.")
print("\nКореляційна матриця:\n", corr.round(3).to_string())
print("\nНайсильніші позитивні зв'язки:")
for a, b, r in positive[:3]:
    print(f"{a} — {b}: r={r:+.3f}")
print("\nНайсильніші негативні зв'язки:")
for a, b, r in negative[:3]:
    print(f"{a} — {b}: r={r:+.3f}")


matplotlib.use("Agg")


columns = [
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Backward_Packets",
    "Fwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Mean",
    "Flow_Bytes_per_s",
    "Flow_Packets_per_s",
    "Flow_IAT_Mean",
]
labels = ["DURATION", "FWD_PKTS", "BWD_PKTS", "FWD_LEN", "BWD_LEN", "BYTES/S", "PKTS/S", "IAT"]

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("cybersecurity_network_traffic_synthetic.csv")
df = pd.read_csv(source)
corr = df[columns].corr(method="pearson")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(9.4, 7.3), dpi=180)
fig.patch.set_facecolor("white")
values = corr.to_numpy()
image = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="none")
ax.set_xticks(range(len(labels)), labels, rotation=43, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(labels)), labels)
ax.tick_params(axis="both", length=0)
ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", left=False, bottom=False)
for i in range(len(columns)):
    for j in range(len(columns)):
        value = float(values[i, j])
        ax.text(j, i, f"{value:+.3f}" if i != j else "1.000", ha="center", va="center", fontsize=9.4, color="white" if abs(value) >= 0.7 else "#15283C", fontweight="bold" if abs(value) > 0.9 else "normal")
colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.04)
colorbar.set_label("Коефіцієнт кореляції Пірсона (r)")
ax.set_title("Кореляційна матриця мережевого трафіку", fontweight="bold", fontsize=14, pad=14)
fig.subplots_adjust(left=0.19, right=0.90, bottom=0.17, top=0.91)
output = Path(__file__).with_name("corelyatsiya_heatmap.png")
fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(output)
