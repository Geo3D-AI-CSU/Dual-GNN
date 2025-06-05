import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 设置中英文字体：SimHei 显示中文，Arial 显示英文字母
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Arial'],
    'axes.unicode_minus': False,
    'font.size': 20,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'figure.dpi': 600,
    'savefig.dpi': 600
})

# 2. 读入并清洗数据
df = pd.read_csv('Plot_CLR_GZ.csv', encoding='utf-8')
for col in ['Ptm3', 'Ptm4', 'Ptm5', 'Ptt']:
    df[col] = df[col].replace('/', np.nan).str.rstrip('%').astype(float)

# 3. 准备画图
methods = df['建模方法'].unique()
strata = ['Ptm3', 'Ptm4', 'Ptm5', 'Ptt']
# colors = ['#2166ac', '#4393c3', '#92c5de', '#d6604d']
colors = ['#f1d8cb', '#fad5bc', '#f7bba1', '#f9d9a2']
n_methods = len(methods)
n_strata = len(strata)

fig, axes = plt.subplots(
    1, n_methods,
    figsize=(6 * n_methods, 6),
    sharey=True
)

# 4. 每个方法一个子图
for ax, method in zip(axes, methods):
    sub = df[df['建模方法'] == method]
    # 对剖面号排序，并取平均(若有重复行)
    pivot = sub.groupby('剖面号')[strata].mean().sort_index()
    profiles = pivot.index.astype(str).tolist()

    x = np.arange(len(profiles))
    width = 0.8 / n_strata

    # 绘制各地层柱
    for i, layer in enumerate(strata):
        vals = pivot[layer].values
        offs = x + i * width - (n_strata - 1) * width / 2
        bars = ax.bar(
            offs, vals, width,
            label=layer,
            color=colors[i],
            edgecolor='black',
            linewidth=0.5
        )
        # 添加百分比标签，横向显示
        # for bar in bars:
        #     h = bar.get_height()
        #     ax.text(
        #         bar.get_x() + bar.get_width() / 2,
        #         h + 0.1,
        #         f'{h:.2f}%',
        #         ha='center', va='bottom',
        #         fontsize=12,
        #         rotation=0
        #     )

    ax.set_title(method, fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.set_xlabel('剖面号', fontsize=20)
    # 只在第一个子图上显示 y 轴标签
    if ax is axes[0]:
        ax.set_ylabel('岩性重合率 (%)', fontsize=20)

# 5. 图例放在右上（整个 axes 区域外）
axes[-1].legend(
    title='地层',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    frameon=True, edgecolor='black'
)

# 6. 网格 & 边框
for ax in axes:
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('method_profile_comparison.png', dpi=600, bbox_inches='tight')
plt.show()
