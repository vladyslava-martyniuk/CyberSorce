import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("cybersecurity_network_traffic_synthetic.csv")

sns.set_theme(style='darkgrid')
plt.figure(figsize=(10, 6))

sns.histplot(data=df, x='Flow_Duration', hue='Label', multiple='dodge', bins=20)

plt.title("Розподіл тривалості потоків (Flow Duration) за типами трафіку")
plt.xlabel("Тривалішсть потоку (Flow Duration)")
plt.ylabel("Кількість")

plt.savefig("traffic_visualization.png")
print("Графік успішно збережено у файл traffic_visualization та готовий до звіту")
plt.show()
