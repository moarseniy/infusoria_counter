import cv2
import numpy as np

# Загрузка исходного изображения
img = cv2.imread('2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if img is None:
    print("Не удалось загрузить изображение.")
    exit()

pattern_size = (6, 6)  # Укажите ваши значения!

# Поиск углов
found, corners = cv2.findChessboardCorners(
    gray,
    pattern_size,
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)

if found:
    # Уточнение координат углов (субпиксельная точность)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Отрисовка углов
    cv2.drawChessboardCorners(img, pattern_size, corners, found)
    cv2.imshow('Chessboard Corners', img)
    cv2.waitKey(0)
else:
    print("Углы не найдены! Проверьте параметры.")
exit(-1)

# Применяем гауссово размытие для уменьшения шума
blur = cv2.GaussianBlur(img, (3, 3), 0)

# Детектор границ Canny
edges = cv2.Canny(blur, 10, 50, apertureSize=3)
cv2.imshow("Edges", edges)
cv2.waitKey(0)

# Дилатация: расширяем линии, чтобы группы линий сливались
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
cv2.imshow("Dilated", dilated)
cv2.waitKey(0)

# Закрытие: соединяем близко расположенные компоненты в единую область
closed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel_dilate, iterations=3)
cv2.imshow("Closed", closed)
cv2.waitKey(0)

kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.erode(closed, kernel_erode, iterations=3)
cv2.imshow("Eroded", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()
