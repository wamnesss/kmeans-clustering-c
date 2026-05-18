import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from sklearn.datasets import make_blobs 

def main():
    print("\n" + "="*50)
    print("НАСТРОЙКА АЛГОРИТМА K-MEANS++ И ДАТАСЕТА")
    print("="*50)
    
    # ввод параметров
    # Пытаемся получить настройки от пользователя через терминал
    try:
        n_points = int(input("➤ Введите количество точек: "))
        k = int(input("➤ Введите количество кластеров: "))
        max_iter = int(input("➤ Введите число итераций: "))
    except ValueError:
        # Если пользователь ввел буквы или нажал Enter, используем безопасные значения по умолчанию
        print("\n[ВНИМАНИЕ] Ошибка ввода. Используются значения по умолчанию: 500, 5, 100.")
        n_points, k, max_iter = 500, 5, 100
    
    print(f"\n[1/3] Генерация датасета из {n_points} точек...")
    
    # генерация датасета
    # make_blobs создает идеальные облака точек.
    # cluster_std=0.5 делает точки внутри облака кучными (плотными).
    # center_box=(-40.0, 40.0) разбрасывает центры облаков по огромному полю, чтобы они не слипались.
    data_x, true_labels = make_blobs(
        n_samples=n_points, 
        centers=k, 
        cluster_std=0.5, 
        center_box=(-40.0, 40.0), 
        random_state=42 # Фиксированный сид для воспроизводимости результатов
    )
    
    # передача данных в си-программу
    # Формируем гигантскую строку (текст), которую мы "скормим" Си-программе
    # Первая строка: количество точек, кластеров и итераций
    input_text = f"{n_points} {k} {max_iter}\n"
    # Дальше идут координаты X и Y всех сгенерированных точек
    for pt in data_x:
        input_text += f"{pt[0]} {pt[1]}\n"
        
    print("[2/3] Запуск математического ядра на Си...")
    
    try:
        # Запускаем наш скомпилированный файл на Си (kmeans_app.exe)
        # input=input_text передает наши точки прямо в функцию scanf() в Си.
        # capture_output=True перехватывает всё, что Си-программа выводит через printf().
        process = subprocess.run(
            ['./kmeans_app.exe'], 
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
            encoding='utf-8' # Защита от проблем с кодировкой в Windows
        )
    except FileNotFoundError:
        print("Ошибка: файл kmeans_app.exe не найден. Скомпилируйте проект!")
        return
    except subprocess.CalledProcessError as e:
        print("Ошибка внутри программы на Си!")
        print(e.stderr) # Если Си-код упал, выводим причину
        return

    # Если Си-программа вывела аналитику (Метод локтя, Силуэт) в поток ошибок (stderr), печатаем её
    if process.stderr:
        print(process.stderr)

    # парсинг ответа от си-кода
    # Списки для хранения результатов, которые прислал Си-алгоритм
    points_x, points_y, predicted_labels = [], [], []
    centroids_x, centroids_y = [], []
    
    # Читаем ответ построчно
    for line in process.stdout.strip().split('\n'):
        parts = line.split()
        if not parts: continue # Пропускаем пустые строки
        
        # Разбираем протокол передачи данных:
        if parts[0] == 'FRAME':
            # Си-код прислал новый шаг (итерацию). Очищаем списки, 
            # чтобы к концу цикла у нас остался только самый финальный результат.
            points_x.clear()
            points_y.clear()
            predicted_labels.clear()
            centroids_x.clear()
            centroids_y.clear()
        elif parts[0] == 'C':
            # 'C' означает Центроид. Сохраняем его координаты X и Y.
            centroids_x.append(float(parts[1]))
            centroids_y.append(float(parts[2]))
        elif parts[0] == 'P':
            # 'P' означает Точка. Сохраняем её координаты и номер кластера, который определил Си.
            points_x.append(float(parts[1]))
            points_y.append(float(parts[2]))
            predicted_labels.append(int(parts[3]))

    # Переводим списки в массивы numpy для быстродействия
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    # выравнивание цветов
    # Алгоритм K-Means находит кластеры, но не знает их "истинных" цветов из эталона.
    # Этот блок перекрашивает кластеры Си-кода в цвета эталона, чтобы графики визуально совпадали.
    aligned_labels = np.zeros_like(predicted_labels)
    for c_id in range(k):
        # Находим все точки текущего кластера
        mask = (predicted_labels == c_id)
        if np.any(mask):
            # Смотрим, какой цвет преобладает у этих же точек в эталоне
            true_colors_in_group = true_labels[mask]
            most_common_color = np.bincount(true_colors_in_group).argmax()
            # Присваиваем всей группе этот правильный цвет
            aligned_labels[mask] = most_common_color

    print("[3/3] Отрисовка графиков сравнения...")
    
    # визуализация
    # Создаем окно с двумя графиками (1 строка, 2 колонки)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Умный выбор палитры: до 20 кластеров - tab20 (различимая), больше - gist_ncar (широкая)
    cmap_name = 'tab20' if k <= 20 else 'gist_ncar'

    # ЛЕВЫЙ ГРАФИК: Ожидание (Истинные метки от генератора)
    ax1.scatter(data_x[:, 0], data_x[:, 1], c=true_labels, cmap=cmap_name, s=40, alpha=0.9, edgecolors='none')
    ax1.set_title(f"Истинные метки (Эталон из make_blobs)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Ось X", fontsize=10)
    ax1.set_ylabel("Ось Y", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ПРАВЫЙ ГРАФИК: Реальность (Результат работы Си-алгоритма)
    ax2.scatter(points_x, points_y, c=aligned_labels, cmap=cmap_name, s=40, alpha=0.9, edgecolors='none')
    # Поверх точек рисуем черные крестики (найденные центроиды)
    ax2.scatter(centroids_x, centroids_y, c='black', marker='X', s=250, edgecolors='white', linewidth=1.5, label='Центроиды')
    ax2.set_title(f"Результат нашего алгоритма K-Means (на Си)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Ось X", fontsize=10)
    ax2.set_ylabel("Ось Y", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right') 

    plt.suptitle(f"Сравнение кластеризации | Количество кластеров $K={k}$", fontsize=18, fontweight='bold', y=0.98)
    
    # Автоматически подгоняем отступы, чтобы графики не слипались
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

if __name__ == '__main__':
    main()