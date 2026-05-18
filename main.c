#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"

// Вспомогательная функция для вывода одного кадра в консоль
void print_frame(Point* data, int n_points, Point* centroids, int k) 
{
    printf("FRAME\n");
    for (int i = 0; i < k; i++) printf("C %f %f\n", centroids[i].x, centroids[i].y);
    for (int i = 0; i < n_points; i++) printf("P %f %f %d\n", data[i].x, data[i].y, data[i].cluster);
}

int main() 
{
    setvbuf(stdout, NULL, _IONBF, 0);
    srand(time(NULL));

    int n_points, k, max_iterations;
    if (scanf("%d %d %d", &n_points, &k, &max_iterations) != 3) return 1;

    Point* data = (Point*)malloc(n_points * sizeof(Point));
    Point* centroids = (Point*)malloc(k * sizeof(Point));

    for (int i = 0; i < n_points; i++) 
    {
        if (scanf("%lf %lf", &data[i].x, &data[i].y) != 2) return 1;
        data[i].cluster = -1;
    }

    int elbow_max = (k > 10) ? k : 10;
    double* elbow_wcss = (double*)malloc(elbow_max * sizeof(double));
    analyze_elbow_method(data, n_points, elbow_max, max_iterations, DIST_EUCLIDEAN, elbow_wcss);

    fprintf(stderr, "\n============================================\n");
    fprintf(stderr, "АНАЛИТИКА: МЕТОД ЛОКТЯ (ELBOW METHOD)\n");
    for (int i = 0; i < elbow_max; i++) fprintf(stderr, " K = %-2d | Инерция (WCSS): %.2f\n", i + 1, elbow_wcss[i]);
    free(elbow_wcss);

    // Инициализация K-Means++
    init_centroids_plus_plus(data, n_points, k, centroids, DIST_EUCLIDEAN);
    
    // Выводим КАДР 0 (Начальное случайное распределение)
    print_frame(data, n_points, centroids, k);

    // Интерактивный пошаговый цикл кластеризации
    int iter = 0;
    int changed = 1;
    while (iter < max_iterations && changed) 
    {
        changed = run_kmeans_one_step(data, n_points, k, centroids, DIST_EUCLIDEAN);
        print_frame(data, n_points, centroids, k); // Выводим кадр после каждого шага
        iter++;
    }

    double silhouette = calculate_silhouette_score(data, n_points, k, DIST_EUCLIDEAN);
    double davies = calculate_davies_bouldin_index(data, n_points, centroids, k, DIST_EUCLIDEAN);
    
    fprintf(stderr, "============================================\n");
    fprintf(stderr, " ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ\n");
    fprintf(stderr, " Силуэтный коэффициент: %.4f (ближе к 1 - лучше)\n", silhouette);
    fprintf(stderr, " Индекс Дэвиса-Болдина: %.4f (ближе к 0 - лучше)\n", davies);
    fprintf(stderr, " Всего итераций до сходимости: %d\n", iter);
    fprintf(stderr, "============================================\n\n");

    free(data);
    free(centroids);
    return 0;
}