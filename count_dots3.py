import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def main():
    image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Не удалось загрузить изображение.")
        return

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 10, 50, apertureSize=3)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    if lines is None:
        print("Линии не обнаружены HoughLines.")
        return

    lines = lines[:, 0, :]  # Приводим к виду (N, 2)
    print("Обнаружено линий:", len(lines))

    X = np.array(lines)
    # Подберите eps экспериментально, например, попробуйте 15
    clustering = DBSCAN(eps=15, min_samples=1).fit(X)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print("Кластер", label, "содержит линий:", count)

    triple_lines = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        # Если кластер содержит ровно 3 линии, то считаем его тройным
        if len(indices) == 3:
            for idx in indices:
                triple_lines.append(lines[idx])
    triple_lines = np.array(triple_lines)
    print("Найдено тройных линий:", len(triple_lines))

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in triple_lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(color_image, pt1, pt2, (0, 0, 255), 2)

    cv2.imshow("Triple Lines", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
