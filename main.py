import matplotlib.pyplot as plt
import numpy as np

SCALE = 2.0

resolutions = ['300x300', '400x400', '500x500', '600x600', '950x950', '2400x2400']
pixels = [90000, 160000, 250000, 360000, 902500, 5760000]

invert_data = {
    'Sequential': [0.0155, 0.0334, 0.0775, 0.2523, 1.3344, 7.9665],
    'OpenMP':     [0.3622, 0.3739, 0.5290, 0.8912, 1.3897, 6.2791],
    'SIMD':       [0.0148, 0.0300, 0.0481, 0.4329, 1.0737, 6.5965],
    'OpenCL':     [0.3712, 0.4646, 0.4417, 0.9628, 2.5910, 11.4874],
}

median_data = {
    'Sequential': [26.7090, 44.1900, 37.1961, 44.1509, 218.9529, 1323.2887],
    'OpenMP':     [4.0899,  4.8766,  5.5758,  5.5720,  19.9564,  113.2147],
    'SIMD':       [1.8800,  2.4666,  2.7516,  2.2350,  9.8878,   47.2766],
    'OpenCL':     [0.3947,  0.3223,  0.5925,  1.0238,  3.1810,   13.0775],
}

sobel_data = {
    'Sequential': [0.6257, 0.6038, 1.2344, 1.5235, 5.0784, 34.5687],
    'OpenMP':     [0.7726, 1.2291, 1.0567, 1.2727, 3.1338, 16.0339],
    'SIMD':       [0.1868, 0.3921, 0.2287, 0.6244, 1.7458, 13.5108],
    'OpenCL':     [0.3074, 0.6273, 0.6850, 1.2410, 2.3729, 13.1175],
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