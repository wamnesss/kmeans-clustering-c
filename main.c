#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

int main() {
    int n, algorithm_choice;
    
    // 1. Читаем заголовок
    if (scanf("%d %d", &n, &algorithm_choice) != 2) return 1;

    int k = 0, max_iter = 0;
    double eps = 0.0;
    int minPts = 0;

    // 2. Читаем параметры в зависимости от выбора (1, 2 или новый режим 3)
    if (algorithm_choice == 1) {
        if (scanf("%d %d", &k, &max_iter) != 2) return 1;
    } else if (algorithm_choice == 2) {
        if (scanf("%lf %d", &eps, &minPts) != 2) return 1;
    } else if (algorithm_choice == 3) {
        // Для режима сравнения считываем все параметры по очереди
        if (scanf("%d %d %lf %d", &k, &max_iter, &eps, &minPts) != 4) return 1;
    }

    // 3. Выделяем память и читаем точки
    Point *points = (Point *)malloc(n * sizeof(Point));
    for (int i = 0; i < n; i++) {
        if (scanf("%lf %lf", &points[i].x, &points[i].y) != 2) return 1;
        points[i].visited = 0; 
        points[i].cluster = -1; 
    }

    // 4. Запускаем нужную математику
    if (algorithm_choice == 1) {
        Point *centroids = (Point *)malloc(k * sizeof(Point));
        init_centroids_plus_plus(points, n, k, centroids, DIST_EUCLIDEAN);
        int actual_iters = run_kmeans(points, n, k, centroids, max_iter, DIST_EUCLIDEAN);
        
        double silhouette = calculate_silhouette_score(points, n, k, DIST_EUCLIDEAN);
        double db_index = calculate_davies_bouldin_index(points, n, centroids, k, DIST_EUCLIDEAN);
        
        fprintf(stderr, "\n============================================\n");
        fprintf(stderr, " ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ (K-MEANS++)\n");
        fprintf(stderr, " Силуэтный коэффициент: %.4f (ближе к 1 - лучше)\n", silhouette);
        fprintf(stderr, " Индекс Дэвиса-Болдина: %.4f (ближе к 0 - лучше)\n", db_index);
        fprintf(stderr, " Всего итераций до сходимости: %d\n", actual_iters);
        fprintf(stderr, "============================================\n");

        printf("FRAME\n");
        for (int j = 0; j < k; j++) printf("C %f %f\n", centroids[j].x, centroids[j].y);
        for (int i = 0; i < n; i++) printf("P %f %f %d\n", points[i].x, points[i].y, points[i].cluster);
        
        free(centroids);
    } 
    else if (algorithm_choice == 2) {
        run_dbscan(points, n, eps, minPts, DIST_EUCLIDEAN);
        
        int found_clusters = 0, noise_count = 0;
        for (int i = 0; i < n; i++) {
            if (points[i].cluster == -1) noise_count++;
            else if (points[i].cluster > found_clusters) found_clusters = points[i].cluster;
        }
        
        fprintf(stderr, "\n============================================\n");
        fprintf(stderr, " ОЦЕНКА РАБОТЫ АЛГОРИТМА (DBSCAN)\n");
        fprintf(stderr, " Найдено кластеров: %d\n", found_clusters);
        fprintf(stderr, " Найдено точек шума (выбросов): %d\n", noise_count);
        fprintf(stderr, "============================================\n");

        printf("FRAME\n");
        for (int i = 0; i < n; i++) printf("P %f %f %d\n", points[i].x, points[i].y, points[i].cluster);
    }
    // режим сравнения
    else if (algorithm_choice == 3) {
        // Создаем отдельный массив для результатов KMeans, чтобы они не перемешались с DBSCAN
        Point *points_kmeans = (Point *)malloc(n * sizeof(Point));
        Point *centroids = (Point *)malloc(k * sizeof(Point));
        
        for (int i = 0; i < n; i++) {
            points_kmeans[i] = points[i]; // Копируем исходные точки
        }

        // Запускаем оба алгоритма параллельно на своих массивах
        init_centroids_plus_plus(points_kmeans, n, k, centroids, DIST_EUCLIDEAN);
        int actual_iters = run_kmeans(points_kmeans, n, k, centroids, max_iter, DIST_EUCLIDEAN);
        run_dbscan(points, n, eps, minPts, DIST_EUCLIDEAN);

        // Считаем и выводим всю аналитику разом в консоль
        double silhouette = calculate_silhouette_score(points_kmeans, n, k, DIST_EUCLIDEAN);
        double db_index = calculate_davies_bouldin_index(points_kmeans, n, centroids, k, DIST_EUCLIDEAN);
        
        int found_clusters = 0, noise_count = 0;
        for (int i = 0; i < n; i++) {
            if (points[i].cluster == -1) noise_count++;
            else if (points[i].cluster > found_clusters) found_clusters = points[i].cluster;
        }

        fprintf(stderr, "\n============================================\n");
        fprintf(stderr, " СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ\n");
        fprintf(stderr, " [K-Means++] Итераций: %d | Силуэт: %.4f | DB-Индекс: %.4f\n", actual_iters, silhouette, db_index);
        fprintf(stderr, " [DBSCAN]    Кластеров: %d | Точек шума: %d\n", found_clusters, noise_count);
        fprintf(stderr, "============================================\n");

        // Выводим специальный маркер для Питона, говорящий, что это двойной режим
        printf("COMPARE_FRAME\n");
        // Сначала шлем центроиды KMeans
        for (int j = 0; j < k; j++) printf("C %f %f\n", centroids[j].x, centroids[j].y);
        // Затем точки KMeans (с префиксом K)
        for (int i = 0; i < n; i++) printf("K %f %f %d\n", points_kmeans[i].x, points_kmeans[i].y, points_kmeans[i].cluster);
        // Затем точки DBSCAN (с префиксом D)
        for (int i = 0; i < n; i++) printf("D %f %f %d\n", points[i].x, points[i].y, points[i].cluster);

        free(centroids);
        free(points_kmeans);
    }

    free(points);
    return 0;
}