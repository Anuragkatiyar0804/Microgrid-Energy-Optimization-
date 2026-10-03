import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/raw/microgrid.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# First 7 days
week = df.iloc[:168]

plt.figure(figsize=(12, 5))
plt.plot(
    week["timestamp"],
    week["load_kw"],
    label="Load"
)
plt.plot(
    week["timestamp"],
    week["solar_kw"],
    label="Solar"
)

plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Load vs Solar")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()