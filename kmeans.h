#ifndef KMEANS_H
#define KMEANS_H

// Перечисление поддерживаемых математических метрик расстояния
typedef enum 
{
    DIST_EUCLIDEAN = 0, // Стандартное Евклидово расстояние
    DIST_MANHATTAN = 1, // Манхэттенское расстояние (сумма модулей)
    DIST_CHEBYSHEV = 2  // Расстояние Чебышева (максимум из модулей)
} DistanceMetric;

// Структура для представления точки в двумерном пространстве
typedef struct 
{
    double x;    
    double y;     
    int cluster; // Для K-Means: номер кластера. Для DBSCAN: 0 = не определен, -1 = Шум, 1, 2... = Кластер
    int visited; // Флаг для DBSCAN: 0 = не посещена, 1 = посещена
} Point;

//БАЗОВЫЕ МАТЕМАТИЧЕСКИЕ ФУНКЦИИ 
double calculate_distance(Point a, Point b, DistanceMetric metric);
double calculate_wcss(Point* data, int n_points, Point* centroids, int k, DistanceMetric metric);
double calculate_silhouette_score(Point* data, int n_points, int k, DistanceMetric metric);
double calculate_davies_bouldin_index(Point* data, int n_points, Point* centroids, int k, DistanceMetric metric);

//ФУНКЦИИ ИНИЦИАЛИЗАЦИИ ЦЕНТРОИДОВ
void init_centroids_random(Point* data, int n_points, int k, Point* centroids);
void init_centroids_plus_plus(Point* data, int n_points, int k, Point* centroids, DistanceMetric metric);

//ЯДРО АЛГОРИТМА КЛАСТЕРИЗАЦИИ (K-MEANS)
int run_kmeans_one_step(Point* data, int n_points, int k, Point* centroids, DistanceMetric metric);
int run_kmeans(Point* data, int n_points, int k, Point* centroids, int max_iterations, DistanceMetric metric);

//ЯДРО АЛГОРИТМА КЛАСТЕРИЗАЦИИ (DBSCAN)
void run_dbscan(Point* data, int n_points, double eps, int minPts, DistanceMetric metric);

//АНАЛИТИЧЕСКИЕ ФУНКЦИИ
void analyze_elbow_method(Point* data, int n_points, int max_k, int max_iterations, DistanceMetric metric, double* wcss_results);

#endif