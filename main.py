import matplotlib.pyplot as plt
import numpy as np

SCALE = 2.0

resolutions = ['300x300', '400x400', '500x500', '600x600', '950x950', '2400x2400']
pixels = [90000, 160000, 250000, 360000, 902500, 5760000]

invert_data = {
    'Sequential': [0.0346, 0.0420, 0.0434, 0.3839, 1.0074, 5.3803],
    'OpenMP':     [0.6078, 0.4911, 0.4350, 0.5095, 1.6469, 7.7196],
    'SIMD':       [0.0172, 0.1136, 0.0690, 0.2911, 0.6726, 5.0472],
    'OpenCL':     [0.4902, 0.5812, 0.6945, 1.4055, 2.6415, 9.2447],
}

median_data = {
    'Sequential': [15.1532, 30.7538, 36.6511, 36.9428, 152.1254, 984.4886],
    'OpenMP':     [5.0534,  6.6268,  5.7356,  5.6820,  47.8455,  117.4008],
    'SIMD':       [0.7909,  4.2538,  2.4504,  3.4945,  7.4681,   47.1407],
    'OpenCL':     [0.5023,  0.8760,  0.5615,  1.1622,  2.1135,   11.7677],
}

sobel_data = {
    'Sequential': [0.6774, 1.3545, 0.9265, 2.0325, 4.0301, 35.6913],
    'OpenMP':     [0.5431, 1.1135, 0.9713, 1.5988, 2.6335, 15.4130],
    'SIMD':       [0.0894, 0.4844, 0.7920, 0.7515, 1.3945, 14.7785],
    'OpenCL':     [0.3579, 0.6363, 0.5409, 1.4300, 2.4181, 11.9788],
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