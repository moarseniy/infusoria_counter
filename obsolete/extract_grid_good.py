import cv2
import numpy as np
import random
from itertools import combinations
import argparse
import os, time, json
import optuna

from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize

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

def preprocess_image2(image_path, params, use_debug):
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

    # Гауссово размытие
    # if params["gauss"]["to_use"]:
    #     img = cv2.GaussianBlur(img, (params["gauss"]["kernel"], params["gauss"]["kernel"]), 0)
    #     if use_debug:
    #         cv2.imshow("GaussianBlur", resize_image_to_fit(img))
    #         cv2.waitKey(0)
        
    if params["adaptive_thresh"]["to_use"]:
        # Адаптивная бинаризация
        img = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, params["adaptive_thresh"]["blockSize"], 
                                   params["adaptive_thresh"]["const"]
        )
        # if use_debug:
        #     cv2.imshow("adaptiveThreshold", resize_image_to_fit(img))
        #     cv2.waitKey(0)

    # img = cv2.bitwise_not(img) 
    # if use_debug:
    #     cv2.imshow("bitwise_not", resize_image_to_fit(img))
    #     cv2.waitKey(0)
    
    # _, img = cv2.threshold(
    #     img, 
    #     40,  # Глобальный порог
    #     255,                            # Максимальное значение
    #     cv2.THRESH_BINARY_INV            # Тип бинаризации
    # )
    # if use_debug:
    #     cv2.imshow("threshold", resize_image_to_fit(img))
    #     cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    cv2.imshow("close", resize_image_to_fit(img))
    cv2.waitKey(0)

    # img = skeletonize(img // 255) 
    # cv2.imshow("skeletonize", resize_image_to_fit(img))
    # cv2.waitKey(0)

    lines = cv2.HoughLinesP(
        img, 
        rho=1, 
        theta=np.pi/180, 
        threshold=params["hough"]["threshold"], 
        minLineLength=params["hough"]["minLineLength"], 
        maxLineGap=params["hough"]["maxLineGap"]
    )
    return img, lines

    # Закрытие: соединяем близко расположенные компоненты в единую область
    # kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_close2, iterations=1)
    # if use_debug:
    #     cv2.imshow("open", resize_image_to_fit(img))
    #     cv2.waitKey(0)

    # img = cv2.medianBlur(img, 3)
    # if use_debug:
    #     cv2.imshow("Median Blur", resize_image_to_fit(img))
    #     cv2.waitKey(0)



    # if params["dilate"]["to_use"]:
    #     # Закрытие: соединяем близко расположенные компоненты в единую область
    #     kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 
    #                                                                3))
    #     img = cv2.dilate(img, kernel_dilate, iterations=1)
    #     if use_debug:
    #         cv2.imshow("dilate", resize_image_to_fit(img))
    #         cv2.waitKey(0)

    # # Детектор границ Canny
    # img = cv2.Canny(img, 90, 150, apertureSize=3)
    # if use_debug:
    #     cv2.imshow("edges", resize_image_to_fit(img))
    #     cv2.waitKey(0)

    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # if use_debug:
    #     cv2.imshow("erode", resize_image_to_fit(img))
    #     cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    if use_debug:
        cv2.imshow("dilate", resize_image_to_fit(img))
        cv2.waitKey(0)


    if params["close"]["to_use"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params["close"]["kernel"], 
                                                            params["close"]["kernel"]))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=params["close"]["iterations"])
        if use_debug:
            cv2.imshow("Morphological Closing", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["findContours"]["to_use"]:
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < params["findContours"]["area"]:
                cv2.drawContours(img, [cnt], 0, 0, -1)
        if use_debug:
            cv2.imshow("findContours", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["dilate"]["to_use"]:
        # Закрытие: соединяем близко расположенные компоненты в единую область
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (params["dilate"]["kernel"], 
                                                                   params["dilate"]["kernel"]))
        img = cv2.dilate(img, kernel_dilate, iterations=params["dilate"]["iterations"])
        if use_debug:
            cv2.imshow("dilate", resize_image_to_fit(img))
            cv2.waitKey(0)

    # if params["erode"]["to_use"]:
    #     kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (params["erode"]["kernel"], 
    #                                                               params["erode"]["kernel"]))
    #     img = cv2.erode(img, kernel_erode, iterations=params["erode"]["iterations"])
    #     if use_debug:
    #         cv2.imshow("erode", resize_image_to_fit(img))
    #         cv2.waitKey(0)
    
    # Детекция линий с оптимизированными параметрами
    lines = cv2.HoughLinesP(
        img, 
        rho=1, 
        theta=np.pi/180, 
        threshold=params["hough"]["threshold"], 
        minLineLength=params["hough"]["minLineLength"], 
        maxLineGap=params["hough"]["maxLineGap"]
    )
    
    print(f"preprocess_image finished\t({time.time() - start_time:.2f} s)")

    return img, lines

def preprocess_image(image_path, params, use_debug):
    start_time = time.time()
    # print("preprocess_image started")
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Не удалось загрузить изображение.")
        exit()

    if params["clahe"]["to_use"]:
        clahe = cv2.createCLAHE(clipLimit=params["clahe"]["clipLimit"], 
                                tileGridSize=(params["clahe"]["tileGridSize"], 
                                              params["clahe"]["tileGridSize"]))
        img = clahe.apply(img)
        if use_debug:
            cv2.imshow("clahe", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["close1"]["to_use"]:
        kernel_close1 = cv2.getStructuringElement(cv2.MORPH_RECT, (params["close1"]["kernel"], 
                                                                   params["close1"]["kernel"]))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close1, iterations=params["close1"]["iterations"])
        if use_debug:
            cv2.imshow("closed1", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["gauss"]["to_use"]:
        # Применяем гауссово размытие для уменьшения шума
        img = cv2.GaussianBlur(img, (params["gauss"]["kernel"], 
                                          params["gauss"]["kernel"]), 0)
        if use_debug:
            cv2.imshow("blur", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["canny"]["to_use"]:
        # Детектор границ Canny
        img = cv2.Canny(img, params["canny"]["left"], 
                                params["canny"]["right"], 
                                apertureSize=3)
        if use_debug:
            cv2.imshow("edges", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["erode1"]["to_use"]:
        kernel_erode1 = cv2.getStructuringElement(cv2.MORPH_RECT, (params["erode1"]["kernel"], 
                                                                   params["erode1"]["kernel"]))
        img = cv2.erode(img, kernel_erode1, iterations=params["erode1"]["iterations"])
        if use_debug:
            cv2.imshow("erode1", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["dilate"]["to_use"]:
        # Дилатация: расширяем линии, чтобы группы линий сливались
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (params["dilate"]["kernel"], 
                                                                   params["dilate"]["kernel"]))
        img = cv2.dilate(img, kernel_dilate, iterations=params["dilate"]["iterations"])
        if use_debug:
            cv2.imshow("dilated", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["close2"]["to_use"]:
        # Закрытие: соединяем близко расположенные компоненты в единую область
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (params["close2"]["kernel"], 
                                                                   params["close2"]["kernel"]))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_close2, iterations=params["close2"]["iterations"])
        if use_debug:
            cv2.imshow("closed", resize_image_to_fit(img))
            cv2.waitKey(0)

    if params["erode2"]["to_use"]:
        # Эрозия: уменьшаем размер линий
        kernel_erode2 = cv2.getStructuringElement(cv2.MORPH_RECT, (params["erode2"]["kernel"], 
                                                                   params["erode2"]["kernel"]))
        img = cv2.erode(img, kernel_erode2, iterations=params["erode2"]["iterations"])
        if use_debug:
            cv2.imshow("erode2", resize_image_to_fit(img))
            cv2.waitKey(0)

    # Поиск линий с использованием преобразования Хафа
    lines = cv2.HoughLinesP(img, rho=1, 
                                    theta=np.pi / 180, 
                                    threshold=params["hough"]["threshold"], 
                                    minLineLength=params["hough"]["minLineLength"], 
                                    maxLineGap=params["hough"]["maxLineGap"])
    print(f"preprocess_image finished\t({time.time() - start_time:.2f} s)")
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

def find_intersections(img, lines, use_debug):
    start_time = time.time() 
    # print("find_intersections started")
    # Находим все пересечения линий
    intersections = []
    if lines is not None and len(lines) and len(lines) < 1000:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                intersection = find_intersection_two_lines(line1, line2)
                if intersection:
                    intersections.append(intersection)

        # Отображаем результат
        if use_debug:
            # Отображаем пересечения на изображении
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for point in intersections:
                cv2.circle(output_img, point, 1, (0, 0, 255), -1)

            cv2.imshow("Intersections", resize_image_to_fit(output_img))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif use_debug and lines is not None:
        print(f"Found {len(lines)} lines!")
    else:
        print("No lines found!")

    # Преобразуем список пересечений в массив NumPy
    intersections = np.array(intersections)

    print(f"find_intersections finished\t({time.time() - start_time:.2f} s)")
    return intersections


# Функция для проверки, образуют ли четыре точки квадрат
def is_square(points, angle_tolerance=30, side_tolerance=0.1):
    # Вычисляем расстояния между всеми парами точек
    dists = [np.linalg.norm(points[i] - points[j]) for i in range(4) for j in range(i + 1, 4)]
    dists.sort()

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

def filter_close_points(points, radius=20):
    filtered_points = []
    for point in points:
        # Проверяем, есть ли уже близкая точка в filtered_points
        if not any(np.linalg.norm(point - fp) < radius for fp in filtered_points):
            filtered_points.append(point)
    return np.array(filtered_points)


def average_close_points(img, points, use_debug=False, radius=20, min_points=2):
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
        
    
    return np.array(averaged_points)

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

def detect_squares(img, params, intersections, use_debug):
    start_time = time.time()
    
    # Применяем фильтрацию
    # filtered_points = filter_close_points(intersections, radius=40)
    # print(f"filter_close_points completed\t({time.time() - start_time:.2f} s)")
    
    # # Отображаем результат
    # if use_debug:
    #     # Отображаем усреднённые точки на изображении
    #     output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     for point in filtered_points:
    #         cv2.circle(output_img, tuple(point), 5, (0, 0, 255), -1)

    #     cv2.imshow("Filtered Intersections", resize_image_to_fit(output_img))
    #     cv2.waitKey(0)

    if len(intersections) == 0:
        return []

    # Применяем усреднение
    start_time = time.time()
    # print("FIRST")
    averaged_points = average_close_points(img, intersections, use_debug, 
                                            radius=params["first_average"]["radius"], 
                                            min_points=params["first_average"]["min_points"])
    if use_debug:
        # Отображаем усреднённые точки на изображении
        output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for point in averaged_points:
            cv2.circle(output_img, tuple(point), 1, (0, 0, 255), -1)

        cv2.imshow("Filtered and Averaged Intersections1", resize_image_to_fit(output_img))
        cv2.waitKey(0)

    if len(averaged_points) > 1000:
        print(f"So many points detected! ({len(averaged_points)})")
        return 1000 * [0]
    elif len(averaged_points) == 0:
        print("No points detected!")
        return []

    # print("SECOND")
    averaged_points = average_close_points(img, averaged_points, use_debug, 
                                            radius=params["second_average"]["radius"], 
                                            min_points=params["second_average"]["min_points"])
    print(f"average_close_points completed\t({time.time() - start_time:.2f} s)")

    # Отображаем результат
    if use_debug:
        # Отображаем усреднённые точки на изображении
        output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for point in averaged_points:
            cv2.circle(output_img, tuple(point), 5, (0, 0, 255), -1)

        cv2.imshow("Filtered and Averaged Intersections2", resize_image_to_fit(output_img))
        cv2.waitKey(0)

    if len(averaged_points) > 300:
        print(f"So many points detected! ({len(averaged_points)})")
        return 300 * [0]
    elif len(averaged_points) == 0:
        print("No points detected!")
        return []

    # Группируем вершины в квадраты
    squares = []
    for comb in combinations(range(len(averaged_points)), 4):  # Перебираем все комбинации из 4 точек
        
        points = np.array([averaged_points[comb[0]], averaged_points[comb[1]], averaged_points[comb[2]], averaged_points[comb[3]]])
        sorted_points = sort_points(points)
        if is_square(sorted_points):
            squares.append(sorted_points)

    filtered_squares = filter_overlapping_squares(squares)

    # Отображаем результат
    if use_debug:
        print(f"Найдено {len(filtered_squares)} квадратов сетки!")
        # Отображаем квадраты на исходном изображении
        output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for square in filtered_squares:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Случайный цвет
            square = square.reshape((-1, 1, 2))
            cv2.polylines(output_img, [square], isClosed=True, color=color, thickness=2)

        cv2.imshow("Detected Squares", resize_image_to_fit(output_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print("detect_squares finished")
    return filtered_squares

def extract_grid(image_path, params):
    use_debug = params["settings"]["debug"]
    img, lines = preprocess_image2(image_path, params["preprocess"], use_debug)

    intersections = find_intersections(img, lines, use_debug)
    return []
    squares = detect_squares(img, params["filter"], intersections, use_debug)

    return squares

def optimize_params(trial, path):
    params = {
        "settings": {
            "debug": False
        },
        "preprocess": {
            "clahe": {
                "to_use": True, #trial.suggest_categorical("clahe_to_use", [True, False]),
                "clipLimit": 2.215, #trial.suggest_float("clipLimit", 1.0, 8.0), 
                "tileGridSize": 8, #trial.suggest_int("tileGridSize", 1, 9)
            },
            "adaptive_thresh": {
                "to_use": True, #trial.suggest_categorical("adaptive_thresh_to_use", [True, False]),
                "blockSize": 11, #trial.suggest_int("blockSize", 3, 21, step=2),
                "const": 10, #trial.suggest_int("const", 1, 15)
            },
            "hough": {
                "threshold": 246, #trial.suggest_int("threshold", 10, 1000),
                "minLineLength": 119, #trial.suggest_int("minLineLength", 10, 700), 
                "maxLineGap": 67 #trial.suggest_int("maxLineGap", 5, 100)
            }
        },
        "filter": {
            "first_average": {
                "radius": trial.suggest_int("radius1", 1, 20),
                "min_points": trial.suggest_int("min_points1", 1, 5)    
            },
            "second_average": {
                "radius": trial.suggest_int("radius2", 1, 40),
                "min_points": trial.suggest_int("min_points2", 2, 10)
            }
        }
    }

    mse_values = []
    for file in sorted(os.listdir(path)):
        grid_count = int(file.split('.')[-2])
        full_path = os.path.join(path, file)
        squares = extract_grid(full_path, params)
        count = len(squares)
        mse = (grid_count - count) ** 2
        mse_values.append(mse)
    return np.mean(mse_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск квадратов на изображении.")
    parser.add_argument("-i", type=str, help="Путь к изображениям.")
    parser.add_argument("-c", type=str, help="Путь к конфигу с настройками.")
    args = parser.parse_args()
    path = args.i
    config = args.c

    # start_time = time.time()
    # study = optuna.create_study(direction="minimize")
    # study.optimize(lambda trial: optimize_params(trial, path), n_trials=100)
    # print(f"Лучшие параметры: {study.best_params}")
    # print(f"Лучшее MSE: {study.best_value}")
    # print(f"Elapsed time:{time.time() - start_time:.2f} s")

    # exit(-1)

    with open(config, "r") as f:
        params = json.load(f)

    mse_values = []
    if os.path.isdir(path):
        for file in sorted(os.listdir(path)):
            grid_count = int(file.split('.')[-2])
            full_path = os.path.join(path, file)
            squares = extract_grid(full_path, params)
            count = len(squares)
            mse = (grid_count - count) ** 2
            mse_values.append(mse)
            print(f"({file})Нужно: {grid_count}, получилось: {count}")
        print(f"MSE:", np.mean(mse_values))
    else:
        squares = extract_grid(args.i, params)