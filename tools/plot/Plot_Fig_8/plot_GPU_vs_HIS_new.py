import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import os
import numpy as np


df = pd.read_csv('all_resultsGPU.csv')

def colors_from_values(values, palette_name):
    normalized = (values - min(values)) / (max(values) - min(values))
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

def annotate_bars(ax):
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = '{:.2f}'.format(p.get_height())
        ax.text(_x, _y, value, ha="center", va="bottom", fontsize=12)


graphs = df['Graph'].unique()

for graph in graphs:
    sub_df = df[df['Graph'] == graph]
    sub_df['speedup'] = sub_df['HIS'] / sub_df['RS']

    plt.figure(figsize=(4, 6))

    #ax = sns.barplot(x='k', y='speedup', data=sub_df, palette=colors_from_values(sub_df['speedup'], "Greens_d"))
    ax = sns.barplot(x='k', y='speedup', data=sub_df, palette=colors_from_values(sub_df['speedup'], sns.dark_palette("#2ec4b6", reverse=True)))
    annotate_bars(ax)

    plt.grid(True)
    plt.xlabel('k', fontsize=22)
    plt.ylabel('Speedup (HIS/RS)', fontsize=22)
    plt.title(f"{graph}", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{graph}_speedup_HIS_new.pdf", format='pdf', dpi=300)
    plt.show()
    plt.clf()