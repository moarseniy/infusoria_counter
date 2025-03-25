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


def update_threshold(_=None):
    # Получаем текущие значения слайдеров
    block_size = cv2.getTrackbarPos("Block Size", "Adaptive Threshold")
    c = cv2.getTrackbarPos("C", "Adaptive Threshold")
    
    # Block Size должен быть нечётным и >= 3
    block_size = max(3, block_size)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    
    # Применяем адаптивную бинаризацию
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c
    )
    
    # Показываем результат
    cv2.imshow("Adaptive Threshold", resize_image_to_fit(adaptive_thresh))

path = "/home/arseniy/infusoria_counter/kontrol_valid_quads/5.1.3.4.jpg"

# Загрузка изображения
img = cv2.imread(path)
if img is None:
    print("Ошибка загрузки изображения!")
    exit()

# Конвертируем в градации серого
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Создаём окно
cv2.namedWindow("Adaptive Threshold")

# Создаём слайдеры
cv2.createTrackbar("Block Size", "Adaptive Threshold", 11, 50, update_threshold)
cv2.createTrackbar("C", "Adaptive Threshold", 2, 20, update_threshold)

# Первоначальная обработка
update_threshold()

# Ожидаем нажатия любой клавиши
cv2.waitKey(0)
cv2.destroyAllWindows()
