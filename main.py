import pandas as pd


df = pd.read_csv("cybersecurity_network_traffic_synthetic.csv")


print("Перші 5 рядків датасету:")
print(df.head())


print("\nЗагальна інформація про датасет:")
df.info()


print("\nНазви стовпців:")
print(df.columns)


print("\nРозмір датасету:")
print(f"Кількість рядків: {df.shape[0]}")
print(f"Кількість стовпців: {df.shape[1]}")