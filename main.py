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

plt.tight_layout()


plt.savefig(
    "traffic_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

print("Графік збережено у файл traffic_visualization.png.")


plt.show()