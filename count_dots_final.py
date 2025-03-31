import cv2
import numpy as np
from functools import reduce
from operator import mul
import os, argparse, json
import csv

from extract_grid_final import extract_grid, resize_image_to_fit

square_colors = [
    (0, 0, 255),   # Красный для квадрата 1
    (0, 0, 0),   # Черный для квадрата 2
    (255, 0, 0),   # Синий для квадрата 3
    (0, 255, 255)  # Желтый для квадрата 4
]

def apply_watershed(orig_img, preprocessed_img):
    img_color = orig_img.copy()

    shape = preprocessed_img.shape

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(preprocessed_img, kernel, iterations=3)

    # Определяем область переднего плана с помощью расстояния
    dist_transform = cv2.distanceTransform(preprocessed_img, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Находим неизвестную область
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Создаем маркеры для Watershed
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # Применяем алгоритм Watershed
    markers = cv2.watershed(img_color, markers)

    # Находим контуры на результате Watershed
    contours = []
    for marker in np.unique(markers):
        if marker > 1:  # 0 и 1 - фон
            mask = np.zeros(shape, dtype="uint8")
            mask[markers == marker] = 255
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours.extend(cnts)

    return contours

def preprocess_image(orig_img, params, use_debug=False):
    
    img = orig_img.copy()
    
    # Применяем размытие
    img = cv2.GaussianBlur(img, (5, 5), 0)
    if use_debug:   
        cv2.imshow("Blurred Image", resize_image_to_fit(img))
        cv2.waitKey(0)

    # Преобразуем изображение в оттенки серого
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применяем инвертированный бинарный порог
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) #95

    # img = cv2.adaptiveThreshold(
    #     img, 255, 
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY_INV, params["adaptive_thresh"]["blockSize"], 
    #                            params["adaptive_thresh"]["const"]
    # )
    if use_debug:
        cv2.imshow("Threshold Image", resize_image_to_fit(img))
        cv2.waitKey(0)

    # Морфологические операции: сначала открытие, затем закрытие
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    if use_debug:   
        cv2.imshow("Opened Image", resize_image_to_fit(img))
        cv2.waitKey(0)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    if use_debug:   
        cv2.imshow("Closed Image", resize_image_to_fit(img))
        cv2.waitKey(0)

    return img

def preprocess_image2(orig_img, params, use_debug=False):
    img = orig_img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=params["clahe"]["clipLimit"], 
    #                         tileGridSize=(params["clahe"]["tileGridSize"], 
    #                                       params["clahe"]["tileGridSize"]))
    # img = clahe.apply(img)
    # if use_debug:
    #     cv2.imshow("clahe", resize_image_to_fit(img))
    #     cv2.waitKey(0)

    # Гауссово размытие
    # if params["gauss"]["to_use"]:
    #     img = cv2.GaussianBlur(img, (params["gauss"]["kernel"], params["gauss"]["kernel"]), 0)
    #     if use_debug:
    #         cv2.imshow("GaussianBlur", resize_image_to_fit(img))
    #         cv2.waitKey(0)
        

    img = cv2.adaptiveThreshold(
        img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, params["adaptive_thresh"]["blockSize"], 
                               params["adaptive_thresh"]["const"]
    )
    if use_debug:
        cv2.imshow("adaptiveThreshold", resize_image_to_fit(img))
        cv2.waitKey(0)

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

    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # if use_debug:
    #     cv2.imshow("close", resize_image_to_fit(img))
    #     cv2.waitKey(0)

    return img

def run(squares, path, params):
    use_debug = params["settings"]["dots_debug"]

    squares[0], squares[1] = squares[1], squares[0] # TODO: fix this dirty hack
    img = cv2.imread(path)
    if img is None:
        print("Ошибка загрузки изображения")
        exit(1)

    preprocessed = preprocess_image(img, params["preprocess"]["dots_detector"], use_debug)

    # contours = apply_watershed(img, preprocessed)

    # Находим контуры
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтруем контуры по площади (пороговые значения подбираются экспериментально)
    filteredContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 30 < area < 1000:
            filteredContours.append(contour)

    # Создаем копию исходного изображения для отрисовки

    # Отрисовываем квадраты сетки для визуализации
    for idx, square in enumerate(squares, start=1):
        cv2.polylines(img, [square], isClosed=True, color=square_colors[idx-1], thickness=2)
        M = cv2.moments(square)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img, f"Square {idx}", (cx, cy),
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
                cv2.drawContours(img, [contour], -1, square_colors[idx], 2)
                # Также можно отметить центр точкой
                cv2.circle(img, point, 3, square_colors[idx], -1)
                break  # Контур учитывается только в одном квадрате

    print(os.path.basename(path))
    # Выводим результаты подсчета для каждого квадрата в консоль
    for idx, count in enumerate(square_counts, start=1):
        print(f"Квадрат {idx}: количество черных точек = {count}")

    # if use_debug:
    cv2.imshow("Filtered Contours with Colored Grid", resize_image_to_fit(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return square_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск точек на изображении.")
    parser.add_argument("-i", type=str, help="Путь к изображениям.")
    parser.add_argument("-c", type=str, help="Путь к конфигу с настройками.")
    args = parser.parse_args()
    path = args.i
    config = args.c

    with open(config, "r") as f:
        params = json.load(f)

    squares = []
    csv_data = []
    idx = 0
    save_csv = False
    if os.path.isdir(path):
        save_csv = True
        for file in sorted(os.listdir(path)):
            full_path = os.path.join(path, file)

            squares = extract_grid(full_path, params)
            square_counts = run(squares, full_path, params)

            for count in square_counts:
                square_number = (idx % 16) + 1  
                csv_data.append({
                    "Название": file, 
                    "Номер квадрата": square_number,
                    "Количество точек": count
                })
                idx += 1

    else:
        squares = extract_grid(args.i, params)
        square_counts = run(squares, path, params)

    # Запись в CSV
    if save_csv:
        with open("results.csv", "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Название", "Номер квадрата", "Количество точек"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_data)
