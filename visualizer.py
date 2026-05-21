import subprocess
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs, make_moons, make_circles

def main():
    print("\n" + "="*50)
    print("НАСТРОЙКА АЛГОРИТМА И ДАТАСЕТА")
    print("="*50)
    
    # 1. ВЫБОР РЕЖИМА
    print("➤ Доступные режимы:")
    print("  1 - Запуск только K-Means++")
    print("  2 - Запуск только DBSCAN")
    print("  3 - СРАВНЕНИЕ АЛГОРИТМОВ БОК О БОК")
    algo_choice = int(input("➤ Выберите режим (1-3): "))
    
    n_points = int(input("➤ Введите количество точек: "))
    
    # 2. ВЫБОР ФОРМЫ (ДОСТУПЕН ДЛЯ ВСЕХ РЕЖИМОВ)
    print("\n➤ Выберите форму данных для тестирования:")
    print("  1 - Сферические кучки (Blobs)")
    print("  2 - Полумесяцы (Moons)")
    print("  3 - Концентрические круги (Circles)")
    print("  4 - Вытянутые полосы (Stretched)")
    shape_choice = int(input("➤ Ваш выбор формы (1-4): "))
    
    # 3. ВВОД ПАРАМЕТРОВ
    if algo_choice == 1:
        k = int(input("➤ Введите количество кластеров K: "))
        max_iter = int(input("➤ Введите число итераций: "))
        eps, min_pts = 0.0, 0
    elif algo_choice == 2:

        eps = float(input("➤ Введите радиус поиска eps: ").replace(',', '.'))
        min_pts = int(input("➤ Введите минимальное число точек minPts: "))
        k, max_iter = 0, 0
    elif algo_choice == 3:
        k = int(input("➤ [K-Means++] Введите количество кластеров K: "))
        max_iter = int(input("➤ [K-Means++] Введите число итераций: "))
        eps = float(input("➤ [DBSCAN] Введите радиус поиска eps: ").replace(',', '.'))
        min_pts = int(input("➤ [DBSCAN] Введите минимальное число точек minPts: "))
            
    print(f"\n[1/3] Генерация датасета из {n_points} точек...")
    
    # --- ГЕНЕРАЦИЯ ВЫБРАННОЙ ФОРМЫ ---
    if shape_choice == 1:
        gen_k = k if algo_choice in [1, 3] else random.randint(3, 5)
        data_x, true_labels = make_blobs(n_samples=n_points, centers=gen_k, cluster_std=0.5, center_box=(-40.0, 40.0), random_state=42)
    elif shape_choice == 2:
        data_x, true_labels = make_moons(n_samples=n_points, noise=0.07, random_state=42)
        data_x *= 20.0
    elif shape_choice == 3:
        data_x, true_labels = make_circles(n_samples=n_points, factor=0.5, noise=0.05, random_state=42)
        data_x *= 20.0
    elif shape_choice == 4:
        gen_k = k if algo_choice in [1, 3] else random.randint(3, 5)
        data_x, true_labels = make_blobs(n_samples=n_points, centers=gen_k, random_state=42)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        data_x = np.dot(data_x, transformation)
        data_x *= 10.0
    else:
        print("Неизвестная форма. Используем кучи.")
        data_x, true_labels = make_blobs(n_samples=n_points, centers=3, random_state=42)

    # --- ПЕРЕДАЧА ДАННЫХ В СИ ---
    if algo_choice == 1:
        input_text = f"{n_points} {algo_choice} {k} {max_iter}\n"
    elif algo_choice == 2:
        input_text = f"{n_points} {algo_choice} {eps} {min_pts}\n"
    elif algo_choice == 3:
        input_text = f"{n_points} {algo_choice} {k} {max_iter} {eps} {min_pts}\n"
        
    for pt in data_x:
        input_text += f"{pt[0]} {pt[1]}\n"
        
    print("[2/3] Запуск математического ядра на Си...")
    
    try:
        process = subprocess.run(
            ['./kmeans_app.exe'], input=input_text, text=True,
            capture_output=True, check=True, encoding='utf-8'
        )
    except FileNotFoundError:
        print("Ошибка: файл kmeans_app.exe не найден. Скомпилируйте проект!")
        return

    if process.stderr:
        print(process.stderr)

    km_points_x, km_points_y, km_labels = [], [], []
    db_points_x, db_points_y, db_labels = [], [], []
    centroids_x, centroids_y = [], []
    
    for line in process.stdout.strip().split('\n'):
        parts = line.split()
        if not parts: continue
        
        if parts[0] == 'COMPARE_FRAME' or parts[0] == 'FRAME':
            km_points_x.clear(); km_points_y.clear(); km_labels.clear()
            db_points_x.clear(); db_points_y.clear(); db_labels.clear()
            centroids_x.clear(); centroids_y.clear()
        elif parts[0] == 'C':
            centroids_x.append(float(parts[1]))
            centroids_y.append(float(parts[2]))
        elif parts[0] == 'P': 
            if algo_choice == 1:
                km_points_x.append(float(parts[1])); km_points_y.append(float(parts[2])); km_labels.append(int(parts[3]))
            else:
                db_points_x.append(float(parts[1])); db_points_y.append(float(parts[2])); db_labels.append(int(parts[3]))
        elif parts[0] == 'K': 
            km_points_x.append(float(parts[1])); km_points_y.append(float(parts[2])); km_labels.append(int(parts[3]))
        elif parts[0] == 'D': 
            db_points_x.append(float(parts[1])); db_points_y.append(float(parts[2])); db_labels.append(int(parts[3]))

    km_labels = np.array(km_labels)
    db_labels = np.array(db_labels)
    true_labels = np.array(true_labels)

    # Выравнивание цветов для K-Means
    km_aligned_labels = np.zeros_like(km_labels) if len(km_labels) > 0 else km_labels
    if algo_choice == 1 or algo_choice == 3:
        for c_id in range(k):
            mask = (km_labels == c_id)
            if np.any(mask):
                true_colors_in_group = true_labels[mask]
                most_common_color = np.bincount(true_colors_in_group).argmax()
                km_aligned_labels[mask] = most_common_color

    print("[3/3] Отрисовка графиков...")
    
    if algo_choice == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        cmap_name = 'tab20'
        
        ax1.scatter(data_x[:, 0], data_x[:, 1], c=true_labels, cmap=cmap_name, s=40, alpha=0.9)
        ax1.set_title("Истинные метки (Эталон)", fontsize=12, fontweight='bold', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        ax2.scatter(km_points_x, km_points_y, c=km_aligned_labels, cmap=cmap_name, s=40, alpha=0.9)
        ax2.scatter(centroids_x, centroids_y, c='black', marker='X', s=200, edgecolors='white', linewidth=1.5, label='Центроиды')
        ax2.set_title("Результат K-Means++ (на Си)", fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')
        
        db_points_x = np.array(db_points_x)
        db_points_y = np.array(db_points_y)
        mask_normal = db_labels != -1
        mask_noise = db_labels == -1
        
        ax3.scatter(db_points_x[mask_normal], db_points_y[mask_normal], c=db_labels[mask_normal], cmap=cmap_name, s=40, alpha=0.9)
        if np.any(mask_noise):
            ax3.scatter(db_points_x[mask_noise], db_points_y[mask_noise], c='black', marker='x', s=30, label='Шум')
        ax3.set_title("Результат DBSCAN (на Си)", fontsize=12, fontweight='bold', pad=10)
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='upper right')
        
        plt.suptitle("Комплексное сравнение алгоритмов кластеризации", fontsize=16, fontweight='bold', y=0.98)
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        num_clusters = k if algo_choice == 1 else len(np.unique(db_labels))
        cmap_name = 'tab20' if num_clusters <= 20 else 'gist_ncar'

        ax1.scatter(data_x[:, 0], data_x[:, 1], c=true_labels, cmap=cmap_name, s=40, alpha=0.9)
        ax1.set_title("Истинные метки (Эталон)", fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, linestyle='--', alpha=0.6)

        if algo_choice == 1:
            ax2.scatter(km_points_x, km_points_y, c=km_aligned_labels, cmap=cmap_name, s=40, alpha=0.9)
            ax2.scatter(centroids_x, centroids_y, c='black', marker='X', s=250, edgecolors='white', linewidth=1.5, label='Центроиды')
        else:
            db_points_x = np.array(db_points_x)
            db_points_y = np.array(db_points_y)
            mask_normal = db_labels != -1
            mask_noise = db_labels == -1
            ax2.scatter(db_points_x[mask_normal], db_points_y[mask_normal], c=db_labels[mask_normal], cmap=cmap_name, s=40, alpha=0.9)
            if np.any(mask_noise):
                ax2.scatter(db_points_x[mask_noise], db_points_y[mask_noise], c='black', marker='x', s=30, label='Шум')
        
        ax2.legend(loc='upper right')
        alg_name = "K-Means++" if algo_choice == 1 else "DBSCAN"
        ax2.set_title(f"Результат нашего алгоритма {alg_name} (на Си)", fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()