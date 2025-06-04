import cv2, json
from extract_grid_good import find_intersections, detect_squares, resize_image_to_fit
import argparse

import numpy as np

def preprocess_image(image_path, params, use_debug):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # CLAHE
    if params["clahe"]["to_use"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        if use_debug:
            cv2.imshow("clahe", resize_image_to_fit(img))
            cv2.waitKey(0)

    # Гауссово размытие
    img = cv2.GaussianBlur(img, (5,5), 0)
    if use_debug:
        cv2.imshow("GaussianBlur", resize_image_to_fit(img))
        cv2.waitKey(0)
    
    # Адаптивная бинаризация
    img = cv2.adaptiveThreshold(
        img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    if use_debug:
        cv2.imshow("adaptiveThreshold", resize_image_to_fit(img))
        cv2.waitKey(0)

    # img = cv2.bitwise_not(img) 
    # if use_debug:
    #     cv2.imshow("bitwise_not", resize_image_to_fit(img))
    #     cv2.waitKey(0)

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 2000:
            cv2.drawContours(img, [cnt], 0, 0, -1)
    if use_debug:
        cv2.imshow("findContours", resize_image_to_fit(img))
        cv2.waitKey(0)

    # Закрытие: соединяем близко расположенные компоненты в единую область
    kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, kernel_close2, iterations=3)
    if use_debug:
        cv2.imshow("dilate", resize_image_to_fit(img))
        cv2.waitKey(0)

    kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.erode(img, kernel_close2, iterations=3)
    if use_debug:
        cv2.imshow("erode", resize_image_to_fit(img))
        cv2.waitKey(0)

    # exit(-1)
    
    # Детекция линий с оптимизированными параметрами
    lines = cv2.HoughLinesP(
        img, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=25, 
        maxLineGap=10
    )
    
    return img, lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск квадратов на изображении.")
    parser.add_argument("-i", type=str, help="Путь к изображениям.")
    parser.add_argument("-c", type=str, help="Путь к конфигу с настройками.")
    args = parser.parse_args()
    path = args.i
    config = args.c

    with open(config, "r") as f:
        params = json.load(f)

    use_debug = params["settings"]["debug"]
    img, lines = preprocess_image(path, params["preprocess"], use_debug)

    intersections = find_intersections(img, lines, use_debug)
    
    squares = detect_squares(img, intersections, use_debug)