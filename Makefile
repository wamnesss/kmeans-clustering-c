CC = gcc
CFLAGS = -Wall -Wextra -std=c99

# Компилируем всё
all: kmeans_app tests_app

# Сборка основной программы
kmeans_app: main.c kmeans.c dbscan.c kmeans.h
	$(CC) $(CFLAGS) main.c kmeans.c dbscan.c -o kmeans_app

# Сборка тестов
tests_app: tests.c kmeans.c dbscan.c kmeans.h
	$(CC) $(CFLAGS) tests.c kmeans.c dbscan.c -o tests_app

# Главная команда: компилируем, запускаем тесты, запускаем Python
run: kmeans_app tests_app
	.\tests_app.exe
	python visualizer.py

# Очистка от скомпилированного мусора
clean:
	del /F /Q kmeans_app.exe tests_app.exe