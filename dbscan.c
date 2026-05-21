#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

void run_dbscan(Point* data, int n_points, double eps, int minPts, DistanceMetric metric) 
{
    int cluster_id = 0; // Счетчик текущего кластера 
    int i, j; 
    
    // Выделяем память под очередь для обхода соседей 
    int* queue = (int*)malloc(n_points * sizeof(int)); 
    // Массив-флаг, чтобы знать, добавлена ли уже точка в очередь 
    int* in_queue = (int*)calloc(n_points, sizeof(int)); 

    // Если метрика Евклидова (возвращает квадрат расстояния),
    // то и радиус поиска (eps) нужно возвести в квадрат для честного сравнения!
    double search_radius = eps;
    if (metric == DIST_EUCLIDEAN) {
        search_radius = eps * eps;
    }

    // Главный цикл: проходимся по всем точкам датасета
    for (i = 0; i < n_points; i++) 
    {
        int neighbor_count; 

        // Если точка уже была проверена ранее, пропускаем её
        if (data[i].visited == 1) continue; 

        data[i].visited = 1; // Помечаем текущую точку как "проверенную"
        neighbor_count = 0; 

        // Ищем всех соседей для текущей точки i
        for (j = 0; j < n_points; j++) 
        {
            // Используем search_radius вместо eps для проверки дистанции
            if (calculate_distance(data[i], data[j], metric) <= search_radius) 
            {
                neighbor_count = neighbor_count + 1; 
            }
        }

        // Если соседей меньше, чем нужно (minPts), точка объявляется Шумом (-1)
        if (neighbor_count < minPts) 
        {
            data[i].cluster = -1; // Шум
        } 
        else 
        {
            // нашли достаточно плотную область. Начинаем новый кластер
            int queue_head = 0; // Начало очереди
            int queue_tail = 0; // Конец очереди

            cluster_id = cluster_id + 1; // Увеличиваем номер кластера
            data[i].cluster = cluster_id; // Присваиваем стартовой точке этот номер

            // Снова пробегаемся по точкам, чтобы добавить всех соседей точки i в очередь на проверку
            for (j = 0; j < n_points; j++) 
            {
                if (i != j) // Саму себя в очередь не добавляем
                {
                    if (calculate_distance(data[i], data[j], metric) <= search_radius) 
                    {
                        queue[queue_tail] = j; // Ставим соседа в конец очереди
                        queue_tail = queue_tail + 1; 
                        in_queue[j] = 1; // Отмечаем, что он уже ждет в очереди
                    }
                }
            }

            // Пока очередь соседей не опустеет, продолжаем растить кластер
            while (queue_head < queue_tail) 
            {
                int curr_p = queue[queue_head]; // Берем первого в очереди
                queue_head = queue_head + 1; // Сдвигаем начало очереди (удаляем элемент)

                // Если эту точку еще не исследовали как самостоятельный центр
                if (data[curr_p].visited == 0) 
                {
                    int curr_neighbors = 0; 
                    int k; 

                    data[curr_p].visited = 1; // Помечаем исследованной

                    // Считаем, сколько соседей уже у точки из очереди
                    for (k = 0; k < n_points; k++) 
                    {
                        if (calculate_distance(data[curr_p], data[k], metric) <= search_radius) 
                        {
                            curr_neighbors = curr_neighbors + 1;
                        }
                    }

                    // Если у нее тоже достаточно соседей, она расширяет кластер дальше
                    if (curr_neighbors >= minPts) 
                    {
                        // Добавляем ЕЁ соседей в общую очередь (если их там еще нет)
                        for (k = 0; k < n_points; k++) 
                        {
                            if (curr_p != k) 
                            {
                                if (calculate_distance(data[curr_p], data[k], metric) <= search_radius) 
                                {
                                    if (in_queue[k] == 0) // Проверка на дубликаты в очереди
                                    {
                                        queue[queue_tail] = k; 
                                        queue_tail = queue_tail + 1; 
                                        in_queue[k] = 1; 
                                    }
                                }
                            }
                        }
                    }
                }

                // Если точка до этого была Шумом (или вообще без кластера), 
                // теперь она официально часть нашего текущего кластера
                if (data[curr_p].cluster <= 0) 
                {
                    data[curr_p].cluster = cluster_id; 
                }
            }
        }
    }

    free(queue); 
    free(in_queue); 
}