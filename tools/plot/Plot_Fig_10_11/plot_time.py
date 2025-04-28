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


plt.figure(figsize=(10, 6))
plt.grid(True, which='both', axis='y')
bars = plt.bar(df_k5["GraphShort"], df_k5["Total"], color="royalblue")
plt.yscale("log")
# plt.xlabel("Graph", fontsize=24)
plt.ylabel("Execution time (ms)", fontsize=24)
# plt.title("Total Execution Time per Graph (k = 5)")
plt.xticks(rotation=45)  # rotate x-axis labels

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:,}', 
             ha='center', va='bottom', fontsize=9, rotation=0)

plt.tight_layout()
plt.savefig("timePlot.pdf", format='pdf', dpi=300)
plt.show()
