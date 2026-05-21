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
// проверяем, что разница между числами меньше микроскопического значения (0.0001).
int is_equal(double a, double b) {
    return fabs(a - b) < 0.0001;
}

// ТЕСТ 1: Проверка математических формул расчета расстояний
void test_distance_metrics() {
    printf("Тест 1: Проверка трех метрик расстояния (Евклид, Манхэттен, Чебышев)... ");
    
    // Создаем две точки: (0, 0) и (3, 4)
    Point p1 = {0.0, 0.0, -1, 0};
    Point c1 = {3.0, 4.0, -1, 0};
    
    // 1. Евклидово расстояние (Квадрат расстояния, чтобы не вычислять медленный корень). 
    // Формула: (3-0)^2 + (4-0)^2 = 9 + 16 = 25.
    assert(is_equal(calculate_distance(p1, c1, DIST_EUCLIDEAN), 25.0)); 
    
    // 2. Манхэттенское расстояние (сумма модулей координат). 
    // Формула: |3-0| + |4-0| = 3 + 4 = 7.
    assert(is_equal(calculate_distance(p1, c1, DIST_MANHATTAN), 7.0));  
    
    // 3. Расстояние Чебышева (максимум из модулей разности координат).
    // Формула: max(|3-0|, |4-0|) = max(3, 4) = 4.
    assert(is_equal(calculate_distance(p1, c1, DIST_CHEBYSHEV), 4.0));  
    
    printf("УСПЕШНО!\n");
}

// ТЕСТ 2: Проверка того, как алгоритм находит центры масс
void test_centroid_update() {
    printf("Тест 2: Проверка логики смещения центроидов K-Means++...\n");
    srand(42); // Фиксируем генератор случайных чисел, чтобы тест всегда проходил одинаково

    // Искусственно создаем 4 точки, явно разделенные на 2 кучки:
    // Первая кучка: (1,1) и (3,3). Очевидно, их центр масс должен стать (2,2)
    // Вторая кучка: (10,10) и (12,12). Их центр масс должен стать (11,11)
    Point data[4] = {
        {1.0, 1.0, -1, 0}, {10.0, 10.0, -1, 0}, 
        {3.0, 3.0, -1, 0}, {12.0, 12.0, -1, 0}  
    };
    
    int n_points = 4;
    int k = 2; // Ищем 2 кластера
    Point centroids[2] = {{0.0, 0.0, -1, 0}, {0.0, 0.0, -1, 0}};
    
    // Запускаем алгоритм
    run_kmeans(data, n_points, k, centroids, 10, DIST_EUCLIDEAN);
    
    // Мы не знаем, какой центроид будет записан в centroids[0], а какой в centroids[1].
    // Поэтому проверяем: один из них должен быть (2,2), а другой (11,11).
    int found_cluster_2 =  (is_equal(centroids[0].x, 2.0) && is_equal(centroids[0].y, 2.0)) ||
                           (is_equal(centroids[1].x, 2.0) && is_equal(centroids[1].y, 2.0));
                           
    int found_cluster_11 = (is_equal(centroids[0].x, 11.0) && is_equal(centroids[0].y, 11.0)) ||
                           (is_equal(centroids[1].x, 11.0) && is_equal(centroids[1].y, 11.0));
    
    assert(found_cluster_2 && found_cluster_11); // Если хотя бы один центр не найден, тест упадет
    printf("  Результат Теста 2: УСПЕШНО!\n");
}

// ТЕСТ 3: Проверка расчета дисперсии (Внутрикластерного расстояния)
void test_elbow_method_logic() {
    printf("Тест 3: Проверка аналитики (Метод локтя WCSS)... ");
    Point data[2] = { {1.0, 1.0, 0, 0}, {2.0, 2.0, 0, 0} };
    double wcss_results[2] = {0.0, 0.0};
    
    // Запускаем расчет "Метода локтя" для данных из двух точек
    analyze_elbow_method(data, 2, 2, 10, DIST_EUCLIDEAN, wcss_results);
    
    // Логика теста: если у нас 2 точки и мы делим их на 2 кластера (K=2), 
    // то каждая точка станет своим собственным центром кластера.
    // Расстояние от точки до самой себя равно нулю, значит общая ошибка (WCSS) ДОЛЖНА БЫТЬ равна 0.0.
    assert(is_equal(wcss_results[1], 0.0));
    printf("УСПЕШНО!\n");
}

int main() {
    // Включаем поддержку UTF-8 в консоли Windows, 
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // Запускаем все три теста по очереди
    printf("\n=== ЗАПУСК МОДУЛЬНЫХ ТЕСТОВ (МАТЕМАТИКА K-MEANS++) ===\n");
    test_distance_metrics();
    test_centroid_update();
    test_elbow_method_logic();
    
    printf("=== ВСЕ ТЕСТЫ ПРОЙДЕНЫ (100%% УСПЕШНО) ===\n\n");
    
    return 0;
}