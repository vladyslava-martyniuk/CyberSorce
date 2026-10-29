from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "cybersecurity_network_traffic_synthetic.csv")

X = df.drop(columns=["Label"])
y = df["Label"].map({"BENIGN": 0, "ATTACK": 1})

if y.isna().any():
    raise ValueError("Невідомі значення у стовпці Label")

y = y.astype(int)
X.to_csv(folder / "X.csv", index=False)
y.to_frame(name="Label").to_csv(folder / "y.csv", index=False)

print("Ознаки X:", X.shape)
print("Цільова змінна y:", y.shape)
print("0 = BENIGN, 1 = ATTACK")
print("Кількість записів у класах:", y.value_counts().sort_index().to_dict())
