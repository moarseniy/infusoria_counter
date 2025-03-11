import cv2
import numpy as np

# Загрузка исходного изображения
img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Не удалось загрузить изображение.")
    exit()

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
