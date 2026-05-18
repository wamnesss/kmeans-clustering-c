#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "kmeans.h"

// Подключаем библиотеку Windows для исправления кодировки
#ifdef _WIN32
#include <windows.h>
#endif

// Помощник для сравнения double
int is_equal(double a, double b) {
    return fabs(a - b) < 0.0001;
}

void test_distance_metrics() {
    printf("Тест 1: Проверка трех метрик расстояния (Евклид, Манхэттен, Чебышев)... ");
    Point p1 = {0.0, 0.0, -1};
    Point c1 = {3.0, 4.0, -1};
    
    assert(is_equal(calculate_distance(p1, c1, DIST_EUCLIDEAN), 25.0)); 
    assert(is_equal(calculate_distance(p1, c1, DIST_MANHATTAN), 7.0));  
    assert(is_equal(calculate_distance(p1, c1, DIST_CHEBYSHEV), 4.0));  
    
    printf("УСПЕШНО!\n");
}

void test_centroid_update() {
    printf("Тест 2: Проверка логики смещения центроидов K-Means++...\n");
    srand(42); 

    Point data[4] = {
        {1.0, 1.0, -1}, {10.0, 10.0, -1}, 
        {3.0, 3.0, -1}, {12.0, 12.0, -1}  
    };
    
    int n_points = 4;
    int k = 2;
    Point centroids[2] = {{0.0, 0.0, -1}, {0.0, 0.0, -1}};
    
    run_kmeans(data, n_points, k, centroids, 10, DIST_EUCLIDEAN);
    
    int found_cluster_2 =  (is_equal(centroids[0].x, 2.0) && is_equal(centroids[0].y, 2.0)) ||
                           (is_equal(centroids[1].x, 2.0) && is_equal(centroids[1].y, 2.0));
                           
    int found_cluster_11 = (is_equal(centroids[0].x, 11.0) && is_equal(centroids[0].y, 11.0)) ||
                           (is_equal(centroids[1].x, 11.0) && is_equal(centroids[1].y, 11.0));
    
    assert(found_cluster_2 && found_cluster_11);
    printf("  Результат Теста 2: УСПЕШНО!\n");
}

void test_elbow_method_logic() {
    printf("Тест 3: Проверка аналитики (Метод локтя WCSS)... ");
    Point data[2] = { {1.0, 1.0, 0}, {2.0, 2.0, 0} };
    double wcss_results[2] = {0.0, 0.0};
    
    analyze_elbow_method(data, 2, 2, 10, DIST_EUCLIDEAN, wcss_results);
    
    assert(is_equal(wcss_results[1], 0.0));
    printf("УСПЕШНО!\n");
}

int main() {
    // Включаем поддержку UTF-8 в консоли Windows
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("\n=== ЗАПУСК МОДУЛЬНЫХ ТЕСТОВ (МАТЕМАТИКА K-MEANS++) ===\n");
    test_distance_metrics();
    test_centroid_update();
    test_elbow_method_logic();
    printf("=== ВСЕ ТЕСТЫ ПРОЙДЕНЫ (100%% УСПЕШНО) ===\n\n");
    
    return 0;
}