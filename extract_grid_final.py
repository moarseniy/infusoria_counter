import os, time, json, random
import cv2
import numpy as np
from itertools import combinations
import argparse
import optuna
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize
from sklearn.metrics import mean_squared_error, mean_absolute_error

from typing import List, Dict, Tuple, Optional
import glob

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

import itertools
import math
from collections import defaultdict
from typing import List, Tuple, Optional, Union


def resize_image_to_fit(image, max_width=1280, max_height=720):
    """
    Масштабирует изображение, чтобы оно поместилось в указанные размеры.
    
    :param image: Исходное изображение.
    :param max_width: Максимальная ширина.
    :param max_height: Максимальная высота.
    :return: Масштабированное изображение.
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def preprocess_image2(image_path, params, print_debug=False, draw_debug=False):
    start_time = time.time()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # if use_debug:
        # cv2.imshow("grayscale", resize_image_to_fit(img))
        # cv2.waitKey(0)
    
    if params["clahe"]["to_use"]:
        clahe = cv2.createCLAHE(clipLimit=params["clahe"]["clipLimit"], 
                                tileGridSize=(params["clahe"]["tileGridSize"], 
                                              params["clahe"]["tileGridSize"]))
        img = clahe.apply(img)
        # if use_debug:
        #     cv2.imshow("clahe", resize_image_to_fit(img))
        #     cv2.waitKey(0)

        
    if params["adaptive_thresh"]["to_use"]:
        # Адаптивная бинаризация
        img = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, params["adaptive_thresh"]["blockSize"], 
                                   params["adaptive_thresh"]["const"]
        )

    if params["close"]["to_use"]:
        kernel = np.ones((params["close"]["kernel"], 
                          params["close"]["kernel"]), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 
                               iterations=params["close"]["iterations"])

    blurred = cv2.GaussianBlur(img, (15, 15), 0)  
    _, img = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)  

    if draw_debug:
        cv2.imshow("Preprocessed image", resize_image_to_fit(img))
        cv2.waitKey(0)

    lines = cv2.HoughLinesP(
        img, 
        rho=1, 
        theta=np.pi/180, 
        threshold=params["hough"]["threshold"], 
        minLineLength=params["hough"]["minLineLength"], 
        maxLineGap=params["hough"]["maxLineGap"]
    )
    return img, lines


# Функция для нахождения пересечения двух линий
def find_intersection_two_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # Линии параллельны

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (int(x), int(y))
    return None

def find_intersections(img, lines, print_debug=False, draw_debug=False):
    start_time = time.time() 
    # print("find_intersections started")
    # Находим все пересечения линий
    intersections = []
    if lines is not None and len(lines) and len(lines) < 1300:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                intersection = find_intersection_two_lines(line1, line2)
                if intersection:
                    intersections.append(intersection)

        # Отображаем результат
        if draw_debug:
            # Отображаем пересечения на изображении
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for point in intersections:
                cv2.circle(output_img, point, 1, (0, 0, 255), -1)

            cv2.imshow("Intersections", resize_image_to_fit(output_img))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif print_debug and lines is not None:
        print(f"Found {len(lines)} lines!")
    elif print_debug:
        print("No lines found!")

    # Преобразуем список пересечений в массив NumPy
    intersections = np.array(intersections)
    # if print_debug:
    #     print(f"find_intersections finished\t({time.time() - start_time:.2f} s)")
    
    return intersections


# Функция для проверки, образуют ли четыре точки квадрат
def is_square(points, angle_tolerance=30, side_tolerance=0.1):
    # print(points)
    # Вычисляем расстояния между всеми парами точек
    dists = [np.linalg.norm(points[i] - points[j]) for i in range(4) for j in range(i + 1, 4)]
    dists.sort()
    # print("DISTS")
    # В квадрате должно быть 4 равные стороны и 2 равные диагонали
    side = dists[0]
    diagonal = dists[-1]

    # Проверяем, что все стороны примерно равны и диагонали примерно равны
    if not (all(abs(d - side) < side_tolerance * side for d in dists[:4]) and
            all(abs(d - diagonal) < side_tolerance * diagonal for d in dists[4:])):
        return False

    # Вычисляем углы между сторонами
    vectors = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        vectors.append(p2 - p1)

    angles = []
    for i in range(4):
        v1 = vectors[i]
        v2 = vectors[(i + 1) % 4]
        dot = np.dot(v1, v2)
        det = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(dot / det) * 180 / np.pi
        angles.append(angle)

    # Проверяем, что углы близки к 90 градусам
    if not all(abs(angle - 90) < angle_tolerance for angle in angles):
        return False

    # Проверяем ориентацию квадрата (наклон не должен превышать 30 градусов)
    # Вычисляем угол наклона первой стороны относительно горизонтали
    v = vectors[0]
    angle_to_horizontal = np.arctan2(v[1], v[0]) * 180 / np.pi
    angle_to_horizontal = abs(angle_to_horizontal % 90)  # Приводим к диапазону [0, 90]
    if angle_to_horizontal > 30 and angle_to_horizontal < 60:  # Исключаем наклон 45 градусов
        return False

    return True

def get_square_area(square):
    """
    Вычисляет площадь квадрата.
    
    :param square: Массив из 4 точек квадрата.
    :return: Площадь квадрата.
    """
    # Используем формулу площади многоугольника (shoelace formula)
    x = square[:, 0]
    y = square[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_square_center(square):
    """Вычисляет центр квадрата как среднее его вершин."""
    return np.mean(square, axis=0)

def is_square_inside(square_small, square_large):
    """Проверяет, находится ли центр маленького квадрата внутри большого и что большой квадрат значительно больше."""
    # Преобразуем большой квадрат в формат контура
    contour = square_large.reshape((-1, 1, 2)).astype(np.int32)
    
    # Вычисляем центр маленького квадрата
    center = get_square_center(square_small)
    
    # Проверяем, находится ли центр внутри большого квадрата
    inside = cv2.pointPolygonTest(contour, tuple(map(int, center)), False) >= 0
    
    # Проверяем, что площадь большого квадрата значительно больше
    area_small = get_square_area(square_small)
    area_large = get_square_area(square_large)
    
    return inside and (area_large > area_small * 1.5)

def filter_overlapping_squares(squares):
    """Удаляет большие квадраты, которые содержат центры маленьких."""
    to_remove = set()
    
    for i in range(len(squares)):
        for j in range(len(squares)):
            if i == j:
                continue
            # Если квадрат i содержит центр квадрата j и больше по площади
            if is_square_inside(squares[j], squares[i]):
                to_remove.add(i)  # Помечаем большой квадрат на удаление
                
    return [square for idx, square in enumerate(squares) if idx not in to_remove]


def average_close_points(img, points, print_debug=False, draw_debug=False, radius=20, min_points=2):
    """
    Кластеризует точки с помощью DBSCAN, усредняет их и отрисовывает на изображении.
    
    :param img: Исходное изображение (в градациях серого).
    :param points: Массив точек формата [[x1, y1], [x2, y2], ...].
    :param radius: Максимальное расстояние между точками в одном кластере (eps в DBSCAN).
    :param min_points: Минимальное количество точек для формирования кластера (min_samples в DBSCAN).
    :param use_debug: Если True, отображает промежуточные результаты.
    :return: Усреднённые точки кластеров.
    """
    # Используем DBSCAN для кластеризации точек
    clustering = DBSCAN(eps=radius, min_samples=min_points).fit(points)
    
    # Получаем метки кластеров
    labels = clustering.labels_
    
    # Усредняем координаты точек в каждом кластере
    averaged_points = []
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Преобразуем изображение в цветное
    
    # Генерация случайных цветов для каждого кластера
    unique_labels = set(labels)
    colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in unique_labels if label != -1}
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Пропускаем шумовые точки (не входящие в кластеры)
        
        # Находим точки, принадлежащие текущему кластеру
        cluster_points = points[labels == cluster_id]

        # Усредняем координаты
        avg_point = np.mean(cluster_points, axis=0).astype(int)
        averaged_points.append(avg_point)
        
        # print(avg_point, len(cluster_points))
        # Выводим информацию о кластере
        # color = colors[cluster_id]
        # print(f"Кластер {cluster_id}: Цвет {color}, Усреднённая точка {avg_point}, Количество точек {len(cluster_points)}")
        
        # if use_debug:
        #     # Отрисовываем точки кластера своим цветом
        #     for point in cluster_points:
        #         cv2.circle(output_img, tuple(point), 3, color, -1)
            
        #     # Отрисовываем усреднённую точку
        #     # cv2.circle(output_img, tuple(avg_point), 5, (0, 255, 0), -1)  # Зелёный цвет для усреднённой точки
        #     # Отображаем результат
        #     cv2.imshow("Filtered and Averaged Intersections", output_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        
    
    return averaged_points

def sort_points(points):
    """
    Сортирует точки в порядке: верхняя левая, верхняя правая, нижняя правая, нижняя левая.
    
    :param points: Массив из четырёх точек [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    :return: Отсортированный массив точек.
    """
    # Сортируем сначала по Y (верхние точки идут первыми), затем по X (левые точки идут первыми)
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))
    
    # Разделяем верхние и нижние точки
    upper_points = sorted_points[:2]
    lower_points = sorted_points[2:]
    
    # Сортируем верхние точки по X (левая, затем правая)
    upper_points = sorted(upper_points, key=lambda p: p[0])
    
    # Сортируем нижние точки по X (правая, затем левая)
    lower_points = sorted(lower_points, key=lambda p: -p[0])
    
    # Объединяем точки в нужном порядке
    return np.array([upper_points[0], upper_points[1], lower_points[0], lower_points[1]])


def find_two_lines_intersection(line1_points, line2_points):
    """
    Находит точку пересечения двух прямых.
    
    Параметры:
    line1_points: tuple of two points [(x1, y1), (x2, y2)] - первая прямая
    line2_points: tuple of two points [(x3, y3), (x4, y4)] - вторая прямая
    
    Возвращает:
    (x, y) - координаты точки пересечения или None, если прямые параллельны
    """
    # Разбираем точки на координаты
    (x1, y1), (x2, y2) = line1_points
    (x3, y3), (x4, y4) = line2_points
    
    # Вычисляем коэффициенты уравнений прямых
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    
    # Решаем систему уравнений
    determinant = A1 * B2 - A2 * B1
    
    if abs(determinant) < 1e-6:  # Прямые параллельны или совпадают
        return None
    
    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    
    return (x, y)

def find_missing_point(points):
    points = np.array(points)
    
    # 1. Сортируем точки по Y (сверху вниз) и X (слева направо)
    sorted_y = points[np.argsort(-points[:, 1])]  # Сортировка по Y (ось направлена вниз)
    sorted_x = points[np.argsort(points[:, 0])]   # Сортировка по X
    
    # 2. Вычисляем средние расстояния между соседями
    y_distances = np.diff(sorted_y[:, 1])
    x_distances = np.diff(sorted_x[:, 0])
    avg_y_dist = np.median(y_distances[y_distances > 0]) if np.any(y_distances > 0) else 0
    avg_x_dist = np.median(x_distances[x_distances > 0]) if np.any(x_distances > 0) else 0
    
    # 3. Строим ожидаемую сетку
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
    
    # Генерируем все возможные точки сетки
    expected_points = []
    for y in np.linspace(max_y, min_y, 3):  # 3 строки (Y вниз)
        for x in np.linspace(min_x, max_x, 3):  # 3 столбца
            expected_points.append([x, y])
    expected_points = np.array(expected_points)
    
    # print(expected_points)

    # 4. Ищем ближайшие реальные точки для каждой ожидаемой
    tree = KDTree(points)
    distances, indices = tree.query(expected_points, k=1)
    # print("distances", distances)
    # print("indices", indices)
    # 5. Находим ожидаемую точку с максимальным расстоянием до ближайшей реальной
    missing_idx = np.argmax(distances)
    # print("missing_idx", missing_idx)
    # 6. Определяем позицию в сетке (строка, столбец)
    row = missing_idx // 3  # Строки 0-2 (0 - верхняя)
    col = missing_idx % 3       # Столбцы 0-2 (0 - левый)
    # print("row, col", row, col)

    missing_point = expected_points[missing_idx]
    # print("MISSING POINT", missing_point)

    row_indices = [row * 3 + 0, row * 3 + 1, row * 3 + 2] # 0,1,2 для row=0; 3,4,5 для row=1 и т.д.    
    row_points = expected_points[row_indices]
    # print("row_points", row_points)
    row_points = row_points[[not np.array_equal(p, missing_point) for p in row_points]]
    # print("row_points", row_points)

    _, row_indices_query = KDTree(points).query(row_points, k=1)
    # print("Closest row: ", row_points, row_indices_query)

    col_indices = [col + 0, col + 3, col + 6]  # 0,3,6 для col=0; 1,4,7 для col=1 и т.д.
    col_points = expected_points[col_indices]
    # print("col_points", col_points)
    col_points = col_points[[not np.array_equal(p, missing_point) for p in col_points]]
    # print("col_points", col_points)

    _, col_indices_query = KDTree(points).query(col_points, k=1)
    # print("Closest col: ", col_points, col_indices_query)
    
    # print("col", points[col_indices_query].tolist())

    # print("row", points[row_indices_query].tolist())
    
    res_missing_point = find_two_lines_intersection(points[col_indices_query].tolist(), 
                                                    points[row_indices_query].tolist())

    return np.array(res_missing_point, dtype=int)


def detect_squares(img, params, intersections, print_debug=False, draw_debug=False, visualize_result=False):
    start_time = time.time()
    
    if len(intersections) == 0:
        return []

    # Применяем усреднение
    start_time = time.time()
    averaged_points = average_close_points(img, intersections, print_debug, draw_debug, 
                                            radius=params["points_filter"]["first_average"]["radius"], 
                                            min_points=params["points_filter"]["first_average"]["min_points"])
    if draw_debug:
        # Отображаем усреднённые точки на изображении
        output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for point in averaged_points:
            cv2.circle(output_img, tuple(point), 5, (0, 0, 255), -1)

        cv2.imshow("Filtered and Averaged Intersections", resize_image_to_fit(output_img))
        cv2.waitKey(0)

    if len(averaged_points) > 100 and print_debug:
        print(f"So many points detected! ({len(averaged_points)})")
        return 100 * [0]
    elif len(averaged_points) == 0 and print_debug:
        print("No points detected!")
        return []
    
    squares_list, points_list = [], []

    # pts = [ (float(pt[0]), float(pt[1])) for pt in averaged_points ]
    # squares_list, points_list = find_chain_of_squares(pts)
    # print(squares_list, points_list)

    # if draw_debug:
    #     # Отображаем квадраты на исходном изображении
    #     output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     for square in squares_list:
    #         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Случайный цвет
    #         square = square.reshape((-1, 1, 2))
    #         cv2.polylines(output_img, [square], isClosed=True, color=color, thickness=2)

    #     cv2.imshow("Temporary detected Squares", resize_image_to_fit(output_img))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    if len(averaged_points) == 8:
        missing_point = find_missing_point(averaged_points)
        averaged_points.append(missing_point)
        
        if print_debug:
            print(f'Found missing point {missing_point}')

        if draw_debug:
            # Отображаем усреднённые точки на изображении
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            for point in averaged_points:
                cv2.circle(output_img, tuple(point), 5, (0, 0, 255), -1)

            cv2.imshow("Found missing point", resize_image_to_fit(output_img))
            cv2.waitKey(0)

        # squares_list, points_list = find_chain_of_squares(points_list)

    averaged_points = np.array(averaged_points)

    # # Группируем вершины в квадраты
    for comb in combinations(range(len(averaged_points)), 4):  # Перебираем все комбинации из 4 точек
        
        points = np.array([averaged_points[comb[0]], averaged_points[comb[1]], averaged_points[comb[2]], averaged_points[comb[3]]])
        sorted_points = sort_points(points)
        if is_square(sorted_points):
            squares_list.append(sorted_points)

    squares_list = filter_overlapping_squares(squares_list)
    
    # if print_debug:
    #     print(f"Найдено {len(squares_list)} квадратов сетки!")

    # print("squares_list", squares_list, visualize_result)

    # Отображаем результат
    if visualize_result:
        # Отображаем квадраты на исходном изображении
        # output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for square in squares_list:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Случайный цвет
            square = square.reshape((-1, 1, 2))
            cv2.polylines(img, [square], isClosed=True, color=color, thickness=2)

        cv2.imshow("Finally detected Squares", resize_image_to_fit(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print("detect_squares finished")
    return squares_list

def sort_squares(squares_list):
    """
    Сортирует квадраты в порядке: правый нижний → правый верхний → левый верхний → левый нижний.
    Критерии:
    1. По убыванию max_x (правые)
    2. По убыванию max_y (нижние)
    3. По возрастанию min_x (левые)
    4. По возрастанию min_y (верхние)
    """
    def sort_key(square):
        x_coords = square[:, 0]
        y_coords = square[:, 1]
        max_x = np.max(x_coords)
        max_y = np.max(y_coords)
        min_x = np.min(x_coords)
        min_y = np.min(y_coords)
        return (-max_x, -max_y, min_x, min_y)
    
    return sorted(squares_list, key=sort_key)

def extract_grid(image_path, params, visualize_result=False):
    print_debug, draw_debug = params["print_debug"], params["draw_debug"]
    img, lines = preprocess_image2(image_path, params["preprocess"], print_debug, draw_debug)

    intersections = find_intersections(img, lines, print_debug, draw_debug)
    # return []
    squares = detect_squares(img, params["postprocess"], intersections, print_debug, draw_debug, visualize_result)

    return sort_squares(squares)

def optimize_params(trial, input_path, output_path):
    params = {
        "settings": {
            "visualize_result": False
        },
        "grid_detector": {
            "debug": False,
            "preprocess": {
                "clahe": {
                    "to_use": False, #trial.suggest_categorical("clahe_to_use", [True, False]),
                    "clipLimit": 2.215, #trial.suggest_float("clipLimit", 1.0, 8.0), 
                    "tileGridSize": 8, #trial.suggest_int("tileGridSize", 1, 9)
                },
                "adaptive_thresh": {
                    "to_use": True, #trial.suggest_categorical("adaptive_thresh_to_use", [True, False]),
                    "blockSize": trial.suggest_int("blockSize", 3, 55, step=2),
                    "const": trial.suggest_int("const", 1, 15)
                },
                "hough": {
                    "threshold": trial.suggest_int("threshold", 10, 1000),
                    "minLineLength": trial.suggest_int("minLineLength", 10, 700), 
                    "maxLineGap": trial.suggest_int("maxLineGap", 5, 100)
                }
            },
            "postprocess": {
                "points_filter": {
                    "first_average": {
                        "radius": 30, # trial.suggest_int("radius1", 1, 20),
                        "min_points": 50
                    }
                }
            }
        }
    }

    metrics = process_image_folder_parallel(input_path, output_path, params["grid_detector"])

    return metrics['MSE']

def find_images(path: str) -> List[str]:
    """Рекурсивно находит все изображения в указанной папке"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() in image_extensions:
            return [path]
        else:
            return []
    
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)

def process_single_image(img_path, grid_params, visualize_result=False):
    """Обрабатывает одно изображение и возвращает результат"""
    try:
        squares = extract_grid(img_path, grid_params, visualize_result)
        result = len(squares)
        img_name = os.path.basename(img_path)
        
        debug_info = None
        if result != 4 and grid_params.get('debug', False):
            debug_info = f"Неверно: {img_path} -> {result}"
        
        return {
            'Название': img_name,
            'Результат': result,
            'Ground_Truth': 4.0,
            'debug_info': debug_info
        }
    except Exception as e:
        return {
            'Название': os.path.basename(img_path),
            'Результат': 1000,
            'Ground_Truth': 4.0,
            'error': str(e)
        }

def process_image_folder(folder_path: str, output_csv: str, grid_params: dict) -> Dict[str, float]:
    """Обрабатывает все изображения в папке и сохраняет результаты"""
    image_files = find_images(folder_path)
    if not image_files:
        print("Изображения не найдены!")
        return pd.DataFrame()
    
    results = []
    for img_path in image_files:
        try:
            squares = extract_grid(img_path, grid_params)
            result = len(squares)

            img_name = os.path.basename(img_path)
            results.append({
                'Название': img_name,
                'Результат': result,
                'Ground_Truth': 4.0
            })
            if result != 4 and grid_params['print_debug']:
                print(f"Неверно: {img_path} -> {result:.4f}")
        except Exception as e:
            print(f"Ошибка обработки {img_path}: {e}")
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    if output_csv:
        # Сохраняем результаты
        results_df.to_csv(output_csv, index=False)
        print(f"\nРезультаты сохранены в: {output_csv}")

    metrics = calculate_metrics(results_df)
    metrics.setdefault('MSE', float('10000'))

    return metrics

def process_image_folder_parallel(folder_path: str, output_csv: str, grid_params: dict) -> Dict[str, float]:
    """Обрабатывает все изображения в папке параллельно и сохраняет результаты"""
    image_files = find_images(folder_path)
    if not image_files:
        print("Изображения не найдены!")
        return {}
    
    results = []
    debug_messages = []
    errors = []
    
    # Создаем частичную функцию с фиксированными параметрами
    process_func = partial(process_single_image, grid_params=grid_params)
    
    # Используем ProcessPoolExecutor для параллельной обработки
    with ProcessPoolExecutor(max_workers=3) as executor:
        # Запускаем задачи
        futures = {executor.submit(process_func, img_path): img_path for img_path in image_files}
        
        # Обрабатываем результаты по мере их появления
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Обработка изображений"):
            img_path = futures[future]
            try:
                result = future.result()
                
                if result['Результат'] == 0 or result['Результат'] >= 20:
                    # Отменяем все ещё незавершённые задачи
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    # Прерываем функцию и возвращаем 1000
                    break 

                results.append(result)
                
                # Собираем отладочные сообщения
                if 'debug_info' in result and result['debug_info']:
                    debug_messages.append(result['debug_info'])
                
                # Собираем ошибки
                if 'error' in result:
                    errors.append(f"Ошибка обработки {img_path}: {result['error']}")
            
            except Exception as e:
                error_msg = f"Неожиданная ошибка при обработке {img_path}: {str(e)}"
                errors.append(error_msg)
                results.append({
                    'Название': os.path.basename(img_path),
                    'Результат': 10000,
                    'Ground_Truth': 4.0,
                    'error': str(e)
                })
    
    executor.shutdown(wait=False)

    # Выводим все отладочные сообщения
    if grid_params.get('debug', False) and debug_messages:
        print("\nОтладочная информация:")
        for msg in debug_messages:
            print(msg)
    
    # Выводим все ошибки
    if errors:
        print("\nОшибки обработки:")
        for error in errors:
            print(error)
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # Удаляем вспомогательные колонки
    results_df = results_df.drop(columns=['debug_info', 'error'], errors='ignore')
    
    if output_csv:
        # Сохраняем результаты
        results_df.to_csv(output_csv, index=False)
        print(f"\nРезультаты сохранены в: {output_csv}")
    
    metrics = calculate_metrics(results_df)
    metrics.setdefault('MSE', float('10000'))

    return metrics

def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    try:
        # Вычисление метрик
        metrics = {
            'MSE': mean_squared_error(results_df['Результат'], results_df['Ground_Truth']),
            'MAE': mean_absolute_error(results_df['Результат'], results_df['Ground_Truth']),
            'RMSE': np.sqrt(mean_squared_error(results_df['Результат'], results_df['Ground_Truth'])),
            'StdDev': np.std(results_df['Результат'] - 4.0)
        }
        
        # Дополнительные метрики
        metrics['Accuracy'] = np.mean(np.isclose(results_df['Результат'], 4.0, atol=0.5))
        
        print("\nМетрики качества (GT=4.0):")
        print("--------------------------")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        with open('metrics.txt', 'w') as f:
            f.write("Метрики качества (Ground Truth=4.0)\n")
            f.write("----------------------------------\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        return metrics
    except Exception as e:
        print(f"Ошибка при расчете метрик: {e}")
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск квадратов на изображении.")
    parser.add_argument("-i", "--input", required=True, type=str, 
                        help="Путь к изображениям.")
    parser.add_argument("-c", "--config", required=False, type=str, 
                        help="Путь к конфигу с настройками.")
    parser.add_argument('-o', '--output', required=False, default='', type=str,
                        help='Путь для сохранения результатов (при обработке папки).')
    args = parser.parse_args()
    input_path = args.input
    config_path = args.config
    output_path = args.output

    if not config_path:
        start_time = time.time()
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: optimize_params(trial, input_path, output_path), n_trials=100)
        print(f"Лучшие параметры: {study.best_params}")
        print(f"Лучшее MSE: {study.best_value}")
        print(f"Elapsed time:{time.time() - start_time:.2f} s")

    else:
        with open(config_path, "r") as f:
            params = json.load(f)

        visualize_result = params["settings"]["visualize_result"]

        # Определяем тип ввода (файл или папка)
        if os.path.isfile(input_path):
            # Обработка одиночного изображения
            result = process_single_image(input_path, params["grid_detector"], visualize_result)

        elif os.path.isdir(input_path):
            # Обработка папки с изображениями
            # result = process_image_folder_parallel(input_path, output_path, params["grid_detector"])
            metrics = process_image_folder(input_path, output_path, params["grid_detector"])
        else:
            print(f"Ошибка: путь не существует или недоступен - {input_path}")
            exit(1)
