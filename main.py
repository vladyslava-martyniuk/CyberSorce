import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

df = pd.read_csv("cybersecurity_network_traffic_synthetic.csv")

def check_data_quality(df):
    """
    Функція для перевірки якості даних у DataFrame.
    df: pandas.DataFrame - вхідний DataFrame для перевірки.
    Повертає DataFrame після перевірки якості даних.
    """
    if not df.empty:
        if df.isnull().sum().any():
            print(f"Кількість пропущених значень у кожному стовпці:\n{df.isnull().sum()}")
        else:
            print("Пропущених значень у даних немає.")

        if df.duplicated().sum() > 0:
            print(f"Кількість дубльованих значень:\n{df.duplicated().sum()}")
        else:
            print("Дубльованих значень у даних немає.")
    else:
        print("DataFrame порожній. Перевірка якості даних неможлива.")
        return df


check_data_quality(df)


def clean_data_quality(df):
    """
    Функція для очищення даних у DataFrame від пропущених та дубльованих значень.
    df: pandas.DataFrame - вхідний DataFrame для очищення.
    Повертає очищений DataFrame після видалення пропущених та дубльованих значень 
    або повідомляє, що очищення не потрібне і повертає оригінальний DataFrame.
    """
    if not df.empty:
        if df.isnull().sum().any() or df.duplicated().sum() > 0:
            df_cleaned = df.dropna().drop_duplicates()
            print(f"Дані до очищення: {df.shape[0]} рядків та {df.shape[1]} стовпців.")
            print(f"Дані після очищення: {df_cleaned.shape[0]} рядків та {df_cleaned.shape[1]} стовпців.")
            return df_cleaned
        else:
            print("Пропущених та дубльованих значень у даних немає. Очищення не потрібне.")
            return df
    else:
        print("DataFrame порожній. Очищення даних неможливе.")
        return df

time.sleep(1)
df_cleaned = clean_data_quality(df)

sns.set_theme(style='darkgrid')
plt.figure(figsize=(10, 6))

sns.histplot(data=df, x='Flow_Duration', hue='Label', multiple='dodge', bins=20)

plt.title("Розподіл тривалості потоків (Flow Duration) за типами трафіку")
plt.xlabel("Тривалішсть потоку (Flow Duration)")
plt.ylabel("Кількість")

plt.savefig("traffic_visualization.png")
print("Графік успішно збережено у файл traffic_visualization та готовий до звіту")
plt.show()
