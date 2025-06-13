import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("data/result.parquet")
sns.scatterplot(data=df, x="x", y="y", alpha=0.5)

plt.savefig("data/plot.png", dpi=300)