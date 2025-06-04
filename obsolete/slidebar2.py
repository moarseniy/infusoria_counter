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


path = "/home/arseniy/infusoria_counter/kontrol_valid_quads/5.1.3.4.jpg"

# Загрузка изображения
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# 1. Инверсия изображения (делаем линии светлее фона)
# inverted = cv2.bitwise_not(img)

# 2. Повышение контраста с помощью CLAHE
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl1 = clahe.apply(inverted)

# 3. Адаптивная бинаризация
# binary = cv2.adaptiveThreshold(
#     cl1,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     21,  # Размер окна
#     4    # Константа C
# )

# 4. Морфологическое закрытие для устранения разрывов
# kernel = np.ones((3,3), np.uint8)
# processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 5. Интерактивная настройка параметров (опционально)
def update(value):
    block_size = cv2.getTrackbarPos('Block Size', 'Result') | 1  # Делаем нечётным
    c = cv2.getTrackbarPos('C', 'Result')
    area = cv2.getTrackbarPos('area', 'Result')
    clipLimit = cv2.getTrackbarPos('clipLimit', 'Result')
    tileGridSize = cv2.getTrackbarPos('tileGridSize', 'Result')

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    img = clahe.apply(image)

    processed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, block_size, c)
    # processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    # processed = cv2.bitwise_not(processed)
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < area:
            cv2.drawContours(processed, [cnt], 0, 0, -1)

    cv2.imshow('Result', resize_image_to_fit(processed))

cv2.namedWindow('Result')
cv2.createTrackbar('Block Size', 'Result', 21, 50, update)
cv2.createTrackbar('C', 'Result', 4, 20, update)
cv2.createTrackbar('area', 'Result', 1, 10000, update)
cv2.createTrackbar('clipLimit', 'Result', 1, 12, update)
cv2.createTrackbar('tileGridSize', 'Result', 1, 12, update)
cv2.waitKey(0)
cv2.destroyAllWindows()
