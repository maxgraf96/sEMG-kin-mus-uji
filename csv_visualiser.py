# %%
import pandas as pd
import seaborn as sns

# Set DPI for plots to 300
sns.set(rc={"figure.dpi": 300})

# Load data
# df = pd.read_csv("data/RawEMG-2023-03-22-13.15.34.csv", skiprows=0, usecols=range(8), dtype=float)
df = pd.read_csv("hu_2022_valdata_sEMG.csv", usecols=range(8), dtype=float)
df = df.iloc[:4000, :]

# %%
# Plot first column against range
sns.lineplot(data=df.iloc[:, 0])