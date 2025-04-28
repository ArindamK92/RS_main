import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("timing_results.csv")
df_k5 = df[df["k"] == 5]


name_mapping = {
    "coauthor": "coauthor",
    "twitter": "twitter",
    "soc-lastfm": "lastfm",
    "soc-pokec": "pokec",
    "patent_graph": "patent",
    "soc-orkut": "orkut",
    "soc-livejournal": "livejournal",
    "soc-sinaweibo": "sinaweibo"
}
plot_order = ["coauthor", "twitter", "lastfm", "pokec", "patent", "orkut", "livejournal", "sinaweibo"]


df_k5["GraphShort"] = df_k5["Graph"].map(name_mapping)
df_k5 = df_k5.dropna(subset=["GraphShort"])
df_k5["GraphShort"] = pd.Categorical(df_k5["GraphShort"], categories=plot_order, ordered=True)
df_k5 = df_k5.sort_values("GraphShort")
df_k5["Step 2"] = df_k5[["Step 2A", "Step 2B", "Step 2C", "Step 2D", "Step 2E", "Step 2F"]].sum(axis=1)
df_k5["% Step 1"] = df_k5["Step 1"] / df_k5["Total"] * 100
df_k5["% Step 2"] = df_k5["Step 2"] / df_k5["Total"] * 100
df_k5["% Step 3"] = df_k5["Step 3"] / df_k5["Total"] * 100


plt.figure(figsize=(15, 6))
x = df_k5["GraphShort"]
p1 = df_k5["% Step 1"]
p2 = df_k5["% Step 2"]
p3 = df_k5["% Step 3"]

plt.bar(x, p1, label="Step 1", color="#0060d6")
plt.bar(x, p2, bottom=p1, label="Step 2", color="#4caf50")
plt.bar(x, p3, bottom=p1 + p2, label="Step 3", color="#f18f01")
plt.ylabel("% of Total Time", fontsize=28)
# plt.title("Step-wise Execution Time Breakdown (as % of Total, k = 5)")
plt.xticks(rotation=45, fontsize=26)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("time_steps.pdf", format='pdf', dpi=300)
plt.show()
