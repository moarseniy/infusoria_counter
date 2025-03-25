import cv2
import numpy as np

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

def main():
    path = "/home/arseniy/infusoria_counter/kontrol_valid_quads/5.1.3.4.jpg"
    # Загрузка изображения
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Ошибка загрузки изображения!")
        return
    
    # Создание окна
    cv2.namedWindow('Grid Enhancement', cv2.WINDOW_NORMAL)
    
    # Создание слайдеров
    cv2.createTrackbar('CLAHE Clip', 'Grid Enhancement', 2, 20, lambda x: None)
    cv2.createTrackbar('CLAHE Tile', 'Grid Enhancement', 8, 20, lambda x: None)
    cv2.createTrackbar('Adapt Thresh', 'Grid Enhancement', 5, 50, lambda x: None)
    cv2.createTrackbar('Morph Size', 'Grid Enhancement', 3, 21, lambda x: None)
    cv2.createTrackbar('Min Area', 'Grid Enhancement', 50, 500, lambda x: None)
    cv2.createTrackbar('Gamma', 'Grid Enhancement', 10, 30, lambda x: None)

    while True:
        # Получение значений слайдеров
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip', 'Grid Enhancement')
        clahe_tile = cv2.getTrackbarPos('CLAHE Tile', 'Grid Enhancement') or 1
        adapt_thresh = cv2.getTrackbarPos('Adapt Thresh', 'Grid Enhancement')
        morph_size = cv2.getTrackbarPos('Morph Size', 'Grid Enhancement') or 1
        min_area = cv2.getTrackbarPos('Min Area', 'Grid Enhancement')
        gamma = cv2.getTrackbarPos('Gamma', 'Grid Enhancement') / 10

        # 1. Гамма-коррекция для выделения сетки
        gamma_corrected = np.power(img/255.0, gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)

        # 2. CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, 
                               tileGridSize=(clahe_tile, clahe_tile))
        clahe_applied = clahe.apply(gamma_corrected)

        # 3. Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(clahe_applied, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, adapt_thresh)

        # 4. Морфологическое закрытие
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 5. Удаление мелких объектов
        contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(morph)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        # 6. Финальная обработка
        result = cv2.bitwise_and(morph, mask)
        result = cv2.merge([result, result, result])

        # Отображение промежуточных этапов
        display = resize_image_to_fit(result)
        cv2.imshow('Grid Enhancement', display)

        # Выход по ESC
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
