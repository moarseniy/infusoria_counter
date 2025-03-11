import cv2
import numpy as np
from functools import reduce
from operator import mul

# Координаты вершин сетки (3x3)
a = [[263, 43], [755, 70], [1245, 99],
     [230, 531], [723, 558], [1230, 590],
     [198, 1024], [693, 1052], [1187, 1077]]
a = np.array(a, dtype=np.int32)

# Определяем 4 квадрата (каждый квадрат – массив вершин в порядке: топ-лево, топ-право, бот-право, бот-лево)
square1 = np.array([a[0], a[1], a[4], a[3]])
square2 = np.array([a[1], a[2], a[5], a[4]])
square3 = np.array([a[3], a[4], a[7], a[6]])
square4 = np.array([a[4], a[5], a[8], a[7]])
squares = [square1, square2, square3, square4]

# Задаем для каждого квадрата свой цвет (B, G, R)
square_colors = [
    (0, 0, 255),   # Красный для квадрата 1
    (0, 255, 0),   # Зеленый для квадрата 2
    (255, 0, 0),   # Синий для квадрата 3
    (0, 255, 255)  # Желтый для квадрата 4
]

# Загружаем изображение
img = cv2.imread("21.jpg")
if img is None:
    print("Ошибка загрузки изображения")
    exit(1)

# Применяем размытие
blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Blurred Image", blur)
cv2.waitKey(0)

# Преобразуем изображение в оттенки серого
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Применяем инвертированный бинарный порог
_, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)

# Морфологические операции: сначала открытие, затем закрытие
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("Opened Image", morph)
cv2.waitKey(0)

morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imshow("Closed Image", morph)
cv2.waitKey(0)

# Находим контуры
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Фильтруем контуры по площади (пороговые значения подбираются экспериментально)
filteredContours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if 35 < area < 1000:
        filteredContours.append(contour)

# Создаем копию исходного изображения для отрисовки
result_img = img.copy()

# Отрисовываем квадраты сетки для визуализации
for idx, square in enumerate(squares, start=1):
    cv2.polylines(result_img, [square], isClosed=True, color=square_colors[idx-1], thickness=2)
    M = cv2.moments(square)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(result_img, f"Square {idx}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Создаем массив для подсчета контуров по квадратам
square_counts = [0, 0, 0, 0]

# Для каждого контура определяем его центроид и проверяем, в каком квадрате он находится
for contour in filteredContours:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    point = (cx, cy)
    
    # Проверяем для каждого квадрата
    for idx, square in enumerate(squares):
        if cv2.pointPolygonTest(square, point, False) >= 0:
            square_counts[idx] += 1
            # Отрисовываем найденный контур в нужном квадрате своим цветом
            cv2.drawContours(result_img, [contour], -1, square_colors[idx], 2)
            # Также можно отметить центр точкой
            cv2.circle(result_img, point, 3, square_colors[idx], -1)
            break  # Контур учитывается только в одном квадрате

# Выводим результаты подсчета для каждого квадрата в консоль
for idx, count in enumerate(square_counts, start=1):
    print(f"Квадрат {idx}: количество черных точек = {count}")

cv2.imshow("Filtered Contours with Colored Grid", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
