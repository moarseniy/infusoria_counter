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

def find_intersections(lines):
    intersections = []
    if lines is not None and len(lines) and len(lines) < 500:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                intersection = find_intersection_two_lines(line1, line2)
                if intersection:
                    intersections.append(intersection)

    elif lines is not None:
        print(f"Found {len(lines)} lines!")
    else:
        print("No lines found!")

    # intersections = np.array(intersections)

    return intersections

def main():
    path = "/home/arseniy/infusoria_counter/kontrol_valid_quads/5.1.3.4.jpg"
    # Загрузка изображения
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Ошибка загрузки изображения!")
        return
    
    # Создание окна
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    
    # Создание слайдеров
    cv2.createTrackbar('Block Size', 'Result', 21, 50, lambda x: None)
    cv2.createTrackbar('C', 'Result', 4, 20, lambda x: None)
    cv2.createTrackbar('close_size', 'Result', 3, 6, lambda x: None)
    cv2.createTrackbar('close_iters', 'Result', 1, 5, lambda x: None)

    while True:
        block_size = cv2.getTrackbarPos('Block Size', 'Result') | 1  # Делаем нечётным
        c = cv2.getTrackbarPos('C', 'Result')
        close_size = cv2.getTrackbarPos('close_size', 'Result') or 1
        close_iters = cv2.getTrackbarPos('close_iters', 'Result')
        
        binary = cv2.adaptiveThreshold(img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, block_size, c)

        kernel = np.ones((close_size, close_size), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

        # cv2.imshow('Result', resize_image_to_fit(morph))

        lines = cv2.HoughLinesP(
            morph, 
            rho=1, 
            theta=np.pi/180, 
            threshold=530, 
            minLineLength=366, 
            maxLineGap=70
        )

        intersections = find_intersections(lines)
        print(f"found{len(intersections)}")
        morph = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        for point in intersections:
            cv2.circle(morph, point, 1, (0, 0, 255), -1)

        cv2.imshow("Result", resize_image_to_fit(morph))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
