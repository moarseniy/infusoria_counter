import cv2
import numpy as np
import random
from itertools import combinations
import argparse

def preprocess_image(image_path):

    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Не удалось загрузить изображение.")
        exit()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    cl1 = clahe.apply(img)
    cv2.imshow("clahe", resize_image_to_fit(cl1))
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opened = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow("opened", resize_image_to_fit(opened))
    cv2.waitKey(0)

    # Применяем гауссово размытие для уменьшения шума
    blur = cv2.GaussianBlur(opened, (3, 3), 0)
    cv2.imshow("blur", resize_image_to_fit(blur))
    cv2.waitKey(0)

    # Детектор границ Canny
    edges = cv2.Canny(blur, 40, 70, apertureSize=3)
    cv2.imshow("edges", resize_image_to_fit(edges))
    cv2.waitKey(0)

    # kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # edges = cv2.erode(edges, kernel_erode, iterations=1)
    # cv2.imshow("erode1", resize_image_to_fit(edges))
    # cv2.waitKey(0)

    # Дилатация: расширяем линии, чтобы группы линий сливались

    dilated = cv2.dilate(edges, kernel, iterations=2)
    cv2.imshow("dilated", resize_image_to_fit(dilated))
    cv2.waitKey(0)

    # Закрытие: соединяем близко расположенные компоненты в единую область
    closed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("closed", resize_image_to_fit(closed))
    cv2.waitKey(0)

    # Эрозия: уменьшаем размер линий
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode = cv2.erode(closed, kernel_erode, iterations=2)
    cv2.imshow("erode2", resize_image_to_fit(erode))
    cv2.waitKey(0)

    # Поиск линий с использованием преобразования Хафа
    lines = cv2.HoughLinesP(erode, 1, np.pi / 180, threshold=500, minLineLength=300, maxLineGap=50)
    return erode, lines
