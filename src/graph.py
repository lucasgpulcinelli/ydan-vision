import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("data/result.parquet")
sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="cluster",
    palette="colorblind",
    s=3,
    edgecolor=None,
    linewidth=0,
)

plt.savefig("data/plot.png", dpi=300)
