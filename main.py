import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
