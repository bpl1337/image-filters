import matplotlib.pyplot as plt
import numpy as np

SCALE = 2.0

resolutions = ['300x300', '400x400', '500x500', '600x600', '950x950', '2400x2400']
pixels = [90000, 160000, 250000, 360000, 902500, 5760000]

invert_data = {
    'Sequential': [0.0256, 0.0581, 0.0740, 0.3761, 1.2611, 7.6297],
    'OpenMP':     [0.4532, 0.3491, 0.6885, 1.2483, 1.3152, 5.7430],
    'SIMD':       [0.0267, 0.0281, 0.0289, 0.3863, 1.1904, 5.7333],
    'OpenCL':     [0.3603, 0.5240, 0.3828, 1.5443, 2.5409, 10.9579],
}

median_data = {
    'Sequential': [23.2204, 33.1446, 34.0246, 35.6081, 198.6419, 1144.8404],
    'OpenMP':     [4.3767,  5.1697,  5.5207,  7.9888,  22.7645,  109.7331],
    'SIMD':       [0.7579,  3.4040,  3.1016,  2.9518,  10.2339,  58.9350],
    'OpenCL':     [0.3401,  0.4588,  0.6845,  1.8001,  2.7854,   9.8373],
}

sobel_data = {
    'Sequential': [0.5162, 1.0819, 1.3082, 2.1527, 4.8588, 41.8887],
    'OpenMP':     [0.7883, 1.2746, 1.2097, 1.9437, 3.0878, 18.7524],
    'SIMD':       [0.0975, 0.1916, 0.4405, 0.7523, 1.6026, 17.0003],
    'OpenCL':     [0.5468, 0.5094, 0.6883, 1.8485, 2.9369, 11.5313],
}

styles = {
    'Sequential': {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'OpenMP':     {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'SIMD':       {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'OpenCL':     {'color': '#1f77b4', 'marker': 'D', 'linestyle': ':'},
}

x_labels = [f"{res}\n({p // 1000}k px)" for res, p in zip(resolutions, pixels)]
x_indexes = np.arange(len(resolutions))


def plot_large_graph(title, data, filename):
    figsize = (12 * SCALE, 8 * SCALE)

    fig, ax = plt.subplots(figsize=figsize)

    for method, values in data.items():
        ax.plot(x_indexes, values, label=method,
                color=styles[method]['color'],
                marker=styles[method]['marker'],
                linestyle=styles[method]['linestyle'],
                linewidth=3, markersize=10)

    ax.set_title(title, fontsize=18 * SCALE, fontweight='bold', pad=20 * SCALE)
    ax.set_xlabel('Разрешение изображения (Кол-во пикселей)', fontsize=14 * SCALE, labelpad=15 * SCALE)
    ax.set_ylabel('Среднее время выполнения (мс) - ЛОГ. ШКАЛА', fontsize=14 * SCALE, labelpad=15 * SCALE)

    ax.set_yscale('log')

    ax.set_xticks(x_indexes)
    ax.set_xticklabels(x_labels, fontsize=12 * SCALE)
    ax.tick_params(axis='y', labelsize=12 * SCALE)

    ax.grid(True, which="major", ls="-", alpha=0.7)
    ax.grid(True, which="minor", ls="--", alpha=0.3)

    ax.legend(fontsize=12 * SCALE, loc='upper left', framealpha=0.95)

    plt.tight_layout()

    scale_suffix = f"_scale{SCALE}x" if SCALE != 1.0 else ""
    output_filename = filename.replace('.png', f'{scale_suffix}.png')

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Сохранено: {output_filename}")
    plt.close()


print(f"\n📊 Генерация графиков с масштабом SCALE = {SCALE}x\n")
plot_large_graph('Инверсия цветов: Зависимость времени от размера', invert_data, 'Graph_1_Invert.png')
plot_large_graph('Медианный фильтр: Зависимость времени от размера', median_data, 'Graph_2_Median.png')
plot_large_graph('Фильтр Собеля: Зависимость времени от размера', sobel_data, 'Graph_3_Sobel.png')
print("\n✅ Все графики готовы!\n")