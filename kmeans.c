#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Функция вычисляет метрику расстояния между двумя точками
double calculate_distance(Point a, Point b, DistanceMetric metric) 
{
    double dx;    
    double dy;     
    double result;

    dx = fabs(a.x - b.x); // Вычисляем модуль разности координат по X
    dy = fabs(a.y - b.y); // Вычисляем модуль разности координат по Y

    switch (metric) // Выбираем формулу в зависимости от переданной метрики
    {
        case DIST_MANHATTAN: // Манхэттенское расстояние
        {
            result = dx + dy; // Просто складываем модули разностей
            break;       
        }
        case DIST_CHEBYSHEV: // Расстояние Чебышева
        {
            if (dx > dy) // Ищем максимальное значение из двух осей
            {
                result = dx; 
            } 
            else 
            {
                result = dy;
            }
            break;
        }
        case DIST_EUCLIDEAN: // Евклидово расстояние (используется по умолчанию)
        default:
        {
            result = (dx * dx) + (dy * dy); // Считаем квадрат расстояния (без корня для скорости)
            break;                
        }
    }

    return result; // Возвращаем вычисленную дистанцию
}

//Функция рассчитывает внутрикластерную сумму расстояний (Инерцию)
double calculate_wcss(Point* data, int n_points, Point* centroids, int k, DistanceMetric metric) 
{
    double total_wcss; // Переменная для накопления общей суммы (инерции)
    int i;// Счетчик для цикла по точкам
    int c;// Переменная для хранения индекса кластера

    total_wcss = 0.0; // Обнуляем сумму перед началом расчета

    for (i = 0; i < n_points; i++) // Проходимся по всем точкам датасета
    {
        c = data[i].cluster; // Узнаем, к какому кластеру привязана текущая точка

        if (c >= 0) // Проверяем, что точка вообще привязана к кластеру
        {
            if (c < k) // Проверяем защиту от выхода за пределы массива центроидов
            {
                total_wcss = total_wcss + calculate_distance(data[i], centroids[c], metric); // Прибавляем расстояние до центра
            }
        }
    }

    return total_wcss; // Возвращаем итоговую инерцию
}

//Функция вычисляет Силуэтный коэффициент кластеризации
double calculate_silhouette_score(Point* data, int n_points, int k, DistanceMetric metric) 
{
    double total_silhouette; // Накопитель суммы коэффициентов всех точек
    int i;// Счетчик текущей точки
    int j;// Счетчик для сравнения с другими точками
    int c;// Счетчик кластеров

    if (k < 2) // Силуэт нельзя посчитать для одного кластера
    {
        return -1.0;
    }

    if (n_points < 2) // Силуэт нельзя посчитать для одной точки
    {
        return -1.0;
    }

    total_silhouette = 0.0; // Начальное значение суммы
    
    for (i = 0; i < n_points; i++) // Запускаем цикл оценки каждой точки
    {
        int my_cluster; // Кластер текущей оцениваемой точки
        double a_i; // Среднее расстояние до своих
        int a_count; // Количество своих соседей
        double b_i; // Минимальное среднее расстояние до чужих
        double s_i; // Индивидуальный силуэт точки

        my_cluster = data[i].cluster; // Запоминаем родной кластер

        if (my_cluster == -1) // Если точка не относится к кластеру
        {
            continue; // Пропускаем её
        }

        a_i = 0.0; // Обнуляем сумму расстояний до своих
        a_count = 0; // Обнуляем счетчик своих

        for (j = 0; j < n_points; j++) // Ищем соседей по своему кластеру
        {
            if (i != j) // Не сравниваем точку саму с собой
            {
                if (data[j].cluster == my_cluster) // Если это точка из нашего кластера
                {
                    a_i = a_i + calculate_distance(data[i], data[j], metric); // Суммируем расстояние
                    a_count = a_count + 1; // Увеличиваем счетчик соседей
                }
            }
        }

        if (a_count > 0) // Если соседи найдены
        {
            a_i = a_i / a_count; // Находим внутрикластерное расстояние
        } 
        else 
        {
            a_i = 0.0; // Если точка одна в кластере, расстояние 0
        }

        b_i = -1.0; // Инициализируем минимальное расстояние до чужих как -1 (флаг)

        for (c = 0; c < k; c++) // Проверяем все существующие кластеры
        {
            double sum_dist; // Сумма расстояний до точек чужого кластера
            int count;// Количество точек в чужом кластере
            double avg_dist; // Среднее расстояние до чужого кластера

            if (c == my_cluster) // Свой кластер мы уже посчитали выше
            {
                continue; // Пропускаем
            }

            sum_dist = 0.0; // Обнуляем для нового проверяемого кластера
            count = 0;      // Обнуляем счетчик

            for (j = 0; j < n_points; j++) // Проходим по всем точкам
            {
                if (data[j].cluster == c) // Если точка принадлежит проверяемому чужому кластеру
                {
                    sum_dist = sum_dist + calculate_distance(data[i], data[j], metric); // Накапливаем расстояние
                    count = count + 1;// Считаем точки
                }
            }

            if (count > 0) // Если в чужом кластере есть точки
            {
                avg_dist = sum_dist / count; // Считаем среднее расстояние до них

                if (b_i == -1.0) // Если это первый проверенный чужой кластер
                {
                    b_i = avg_dist; // Запоминаем его как минимальное
                } 
                else if (avg_dist < b_i) // Если нашли кластер ближе, чем предыдущий
                {
                    b_i = avg_dist; // Обновляем минимальное расстояние
                }
            }
        }
        
        s_i = 0.0; // Базовое значение индивидуального силуэта

        if (a_count > 0) // Если есть с чем сравнивать внутри
        {
            if (b_i != -1.0) // Если есть с чем сравнивать снаружи
            {
                double max_ab; // Знаменатель формулы силуэта 

                if (a_i > b_i) // Если внутрикластерное больше
                {
                    max_ab = a_i; // Берем его
                } 
                else 
                {
                    max_ab = b_i; // Берем межкластерное
                }

                if (max_ab > 0) // Защита от деления на ноль
                {
                    s_i = (b_i - a_i) / max_ab; // Формула вычисления Силуэта
                }
            }
        }

        total_silhouette = total_silhouette + s_i; // Добавляем силуэт точки к общей сумме
    }

    return total_silhouette / n_points; // Возвращаем средний силуэт по всей выборке
}

//Функция вычисляет Индекс Дэвиса-Болдина.
double calculate_davies_bouldin_index(Point* data, int n_points, Point* centroids, int k, DistanceMetric metric) 
{
    double* s_i; // Массив средних внутрикластерных расстояний
    int* counts; // Массив количества точек в кластерах
    int i; 
    int j;
    int c;
    double db_index; // Итоговый индекс

    if (k < 2) // минимум два кластера для сравнения
    {
        return -1.0;
    }
    
    s_i = (double*)calloc(k, sizeof(double)); // Выделяем чистую память под расстояния
    counts = (int*)calloc(k, sizeof(int)); // Выделяем чистую память под счетчики
    
    for (i = 0; i < n_points; i++) // Собираем статистику по точкам
    {
        c = data[i].cluster; // Получаем кластер точки

        if (c >= 0) // Если точка распределена
        {
            if (c < k) // Проверка безопасности индексов
            {
                s_i[c] = s_i[c] + calculate_distance(data[i], centroids[c], metric); // Суммируем расстояние до центра
                counts[c] = counts[c] + 1;  // Увеличиваем размер кластера
            }
        }
    }
    
    for (c = 0; c < k; c++) // Проходим по всем кластерам
    {
        if (counts[c] > 0) // Если кластер не пустой
        {
            s_i[c] = sqrt(s_i[c] / counts[c]); // Вычисляем среднеквадратичное отклонение
        }
    }
    
    db_index = 0.0; // Инициализируем итоговый индекс

    for (i = 0; i < k; i++) // Сравниваем каждый кластер
    {
        double max_ratio; // Максимальное отношение
        max_ratio = 0.0;  // Начальное значение

        for (j = 0; j < k; j++) // с каждым другим кластером
        {
            double d_ij;  // Расстояние между центрами двух кластеров
            double ratio; // Значение отношения

            if (i == j) // Не сравниваем кластер с самим собой
            {
                continue; // Пропускаем итерацию
            }

            d_ij = sqrt(calculate_distance(centroids[i], centroids[j], metric)); // Измеряем дистанцию между центроидами

            if (d_ij > 0) // Если центры не слиплись в одной точке
            {
                ratio = (s_i[i] + s_i[j]) / d_ij; // Формула отношения разброса к расстоянию

                if (ratio > max_ratio) // Ищем самое большое отношение
                {
                    max_ratio = ratio; // Обновляем максимум
                }
            }
        }

        db_index = db_index + max_ratio; // Добавляем наихудший показатель кластера к общему индексу
    }
    
    free(s_i); // Освобождаем память массива отклонений
    free(counts); // Освобождаем память массива счетчиков

    return db_index / k; // Возвращаем усредненный индекс
}

void init_centroids_random(Point* data, int n_points, int k, Point* centroids) 
{
    int i; // Счетчик циклов
    int idx; // Случайный индекс точки

    for (i = 0; i < k; i++) // Надо найти K стартовых позиций
    {
        idx = rand() % n_points;       // Генерируем случайный номер от 0 до количества точек
        centroids[i].x = data[idx].x;  // Копируем координату X случайной точки
        centroids[i].y = data[idx].y;  // Копируем координату Y случайной точки
    }
}

//Функция инициализации алгоритма K-Means.
void init_centroids_plus_plus(Point* data, int n_points, int k, Point* centroids, DistanceMetric metric) 
{
    int first_idx;// Индекс для самого первого центроида
    double* dist_array; // Массив минимальных расстояний от точек до центров
    int c; // Счетчик текущего назначаемого центроида
    int i; // Счетчик точек
    int j; // Счетчик уже выбранных центроидов

    first_idx = rand() % n_points; // Первый центр выбираем абсолютно случайно
    centroids[0].x = data[first_idx].x; // Присваиваем X первой точки
    centroids[0].y = data[first_idx].y; // Присваиваем Y первой точки
    
    dist_array = (double*)malloc(n_points * sizeof(double)); // Выделяем память под массив дистанций

    for (c = 1; c < k; c++) // Цикл поиска оставшихся K-1 центроидов
    {
        double total_sum; // Сумма всех минимальных расстояний
        double target; // Случайное значение-цель на рулетке
        double current_sum; // Текущая сумма при кручении рулетки
        int chosen_idx; // Индекс точки, которая станет новым центром

        total_sum = 0.0; // Обнуляем сумму для нового раунда

        for (i = 0; i < n_points; i++) // Вычисляем расстояние от каждой точки до ближайшего центра
        {
            double min_dist; // Минимальная дистанция для текущей точки
            
            min_dist = -1.0; // флаг -1

            for (j = 0; j < c; j++) // Проверяем все уже зафиксированные центры
            {
                double current_dist;

                current_dist = calculate_distance(data[i], centroids[j], metric); // Считаем дистанцию

                if (min_dist == -1.0) // Если это первый проверенный центр
                {
                    min_dist = current_dist; // Берем его дистанцию как минимальную
                }
                else if (current_dist < min_dist) 
                {
                    min_dist = current_dist; 
                }
            }

            dist_array[i] = min_dist; // Сохраняем минимальную дистанцию в массив
            total_sum = total_sum + min_dist; // Прибавляем к общей сумме
        }

        target = ((double)rand() / RAND_MAX) * total_sum; // Бросаем шарик на рулетку
        current_sum = 0.0; // Начинаем крутить рулетку с нуля
        chosen_idx = n_points - 1;  // По умолчанию выбираем последнюю точку
        
        for (i = 0; i < n_points; i++) // Проверяем сектора рулетки
        {
            current_sum = current_sum + dist_array[i]; // Прибавляем ширину сектора текущей точки

            if (current_sum >= target) // Если шарик упал в этот сектор
            { 
                chosen_idx = i; // Назначаем эту точку победителем
                break;  
            }
        }

        centroids[c].x = data[chosen_idx].x; 
        centroids[c].y = data[chosen_idx].y;
    }

    free(dist_array); 
}

//Функция выполняет строго одну математическую итерацию алгоритма
int run_kmeans_one_step(Point* data, int n_points, int k, Point* centroids, DistanceMetric metric) 
{
    int changed;// Флаг изменения кластеров 
    int i;     
    int j;    
    int c;      
    double* sum_x; 
    double* sum_y; 
    int* counts;   // Массив количества точек в кластерах

    changed = 0; // Изначально предполагаем, что изменений не будет

    for (i = 0; i < n_points; i++) // ЭТАП 1: Распределение точек по кластерам
    {
        double min_dist; // Наименьшее найденное расстояние
        int best_cluster; // Индекс самого близкого кластера

        min_dist = -1.0;  // Флаг расстояние не найдено
        best_cluster = -1;// Флаг кластер не найден

        for (j = 0; j < k; j++) // Проверяем расстояние до каждого центроида
        {
            double current_dist; // Расстояние до текущего проверяемого центроида

            current_dist = calculate_distance(data[i], centroids[j], metric); // Расчет расстояния

            if (min_dist == -1.0) // Если это первый проверенный центроид
            {
                min_dist = current_dist; // Берем его за эталон
                best_cluster = j; // Запоминаем его индекс
            }
            else if (current_dist < min_dist) // Если нашли центроид ближе эталона
            {
                min_dist = current_dist; // Обновляем эталон
                best_cluster = j; // Обновляем индекс лучшего кластера
            }
        }

        if (data[i].cluster != best_cluster) // Если точка сменила свою координату
        {
            data[i].cluster = best_cluster; // Обновляем координату точки
            changed = 1; // Фиксируем факт движения алгоритма
        }
    }

    sum_x = (double*)calloc(k, sizeof(double));
    sum_y = (double*)calloc(k, sizeof(double));
    counts = (int*)calloc(k, sizeof(int)); 

    for (i = 0; i < n_points; i++) // ЭТАП 2: Сбор данных для пересчета центров
    {
        c = data[i].cluster; // Узнаем, куда теперь принадлежит точка

        if (c != -1) // Если точка не потерялась
        {
            sum_x[c] = sum_x[c] + data[i].x; 
            sum_y[c] = sum_y[c] + data[i].y; 
            counts[c] = counts[c] + 1;
        }
    }
    
    for (j = 0; j < k; j++) // ЭТАП 3: Физическое смещение центроидов
    {
        if (counts[j] > 0) // Если кластер не опустел
        {
            centroids[j].x = sum_x[j] / counts[j]; 
            centroids[j].y = sum_y[j] / counts[j]; 
        }
    }

    free(sum_x);  
    free(sum_y);  
    free(counts);

    return changed; // Возвращаем статус (1 - работаем дальше, 0 - алгоритм сошелся)
}

//Функция: цикл кластеризации.
int run_kmeans(Point* data, int n_points, int k, Point* centroids, int max_iterations, DistanceMetric metric) 
{
    int iter; // Счетчик выполненных итераций
    int changed; // Флаг, показывающий, двигались ли точки

    iter = 0; // Начинаем с нулевой итерации
    changed = 1; // Задаем 1, чтобы войти в цикл

    while (iter < max_iterations) // Крутим цикл, пока не достигнем лимита
    {
        if (changed != 0) // Проверяем, были ли изменения на прошлом шаге
        {
            changed = run_kmeans_one_step(data, n_points, k, centroids, metric); // Делаем шаг алгоритма
            iter = iter + 1;                                                   
        }
        else 
        {
            break;
        }
    }

    return iter; // Возвращаем количество затраченных итераций
}

//Реализация метода локтя
void analyze_elbow_method(Point* data, int n_points, int max_k, int max_iterations, DistanceMetric metric, double* wcss_results) 
{
    Point* temp_data; // Временный массив точек
    int k_test;// Количество кластеров в текущем тесте
    int i; // Счетчик точек

    temp_data = (Point*)malloc(n_points * sizeof(Point)); 

    for (k_test = 1; k_test <= max_k; k_test++) // Тестируем K от 1 до максимума
    {
        Point* temp_centroids; // Массив центроидов для текущего теста

        for (i = 0; i < n_points; i++) // Копируем координаты
        {
            temp_data[i].x = data[i].x;    
            temp_data[i].y = data[i].y;  
            temp_data[i].cluster = -1; // Сбрасываем кластеры в "неопределено"
        }

        temp_centroids = (Point*)malloc(k_test * sizeof(Point)); // Создаем временные центры

        init_centroids_plus_plus(temp_data, n_points, k_test, temp_centroids, metric); // Умная инициализация
        
        run_kmeans(temp_data, n_points, k_test, temp_centroids, max_iterations, metric); // Запуск алгоритма
        
        wcss_results[k_test - 1] = calculate_wcss(temp_data, n_points, temp_centroids, k_test, metric); // Запись метрики в отчет
        
        free(temp_centroids);
    }

    free(temp_data);
}