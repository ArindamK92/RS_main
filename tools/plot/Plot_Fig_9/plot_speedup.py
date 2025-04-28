import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import os


df = pd.read_csv('all_resultsGPU.csv')
df_k5 = df[df['k'] == 5].copy()

# Compute speedups
df_k5['Speedup_HIS_RS'] = df_k5['HIS'] / df_k5['RS']
df_k5['Speedup_ABC_RS'] = df_k5['ABC'] / df_k5['RS']


sns.set(style='whitegrid')


for graph in df_k5['Graph'].unique():
    sub_df = df_k5[df_k5['Graph'] == graph]

    plot_df = pd.DataFrame({
        'Metric': ['HIS/RS', 'ABC/RS'],
        'Speedup': [sub_df['Speedup_HIS_RS'].values[0], sub_df['Speedup_ABC_RS'].values[0]]
    })

    plt.figure(figsize=(3, 5))
    ax = sns.barplot(x='Metric', y='Speedup', data=plot_df, palette='rocket_r')

    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_height()
        ax.text(_x, _y + 0.02 * _y, f'{_y:.2f}', ha='center', va='bottom', fontsize=12)

    plt.title(f'{graph}', fontsize=16)
    plt.ylabel('Speedup', fontsize=18)
    plt.xlabel('')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    filename = f"{graph}_k5_speedup.pdf"
    plt.savefig(filename, format='pdf', dpi=300)
    plt.show()
    plt.clf()
