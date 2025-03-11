import cv2
import numpy as np

def get_intersections(verticals, horizontals):
    """
    Находит пересечения между списками вертикальных и горизонтальных линий.
    Каждая линия представлена как (x1,y1,x2,y2). Предполагается, что вертикальные линии почти вертикальны,
    а горизонтальные почти горизонтальны.
    """
    intersections = []
    for vx in verticals:
        # Для вертикальной линии возьмём среднее значение x
        x = (vx[0] + vx[2]) / 2.0
        for hy in horizontals:
            # Для горизонтальной линии возьмём среднее значение y
            y = (hy[1] + hy[3]) / 2.0
            intersections.append((int(x), int(y)))
    return intersections

def cluster_lines(lines, axis=0, thresh=20):
    """
    Кластеризуем линии по средней координате (x для вертикальных, y для горизонтальных).
    Возвращаем список усреднённых координат.
    """
    coords = []
    for line in lines:
        if axis == 0:
            # Вертикальные: берем среднее x
            coords.append((line[0] + line[2]) / 2.0)
        else:
            # Горизонтальные: берем среднее y
            coords.append((line[1] + line[3]) / 2.0)
    coords = sorted(coords)
    clusters = []
    current_cluster = [coords[0]]
    for c in coords[1:]:
        if abs(c - current_cluster[-1]) < thresh:
            current_cluster.append(c)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [c]
    clusters.append(np.mean(current_cluster))
    return clusters

# Загружаем изображение
img = cv2.imread("2.jpg")
if img is None:
    print("Ошибка загрузки изображения")
    exit(1)

# Предобработка: перевод в оттенки серого, нормализация яркости
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

cv2.imshow("Detected Lines", gray)
cv2.waitKey(0)

# Применяем гауссово размытие для подавления шума
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Выделяем границы с помощью Canny (настройте пороги под ваше изображение)
edges = cv2.Canny(blur, 50, 150, apertureSize=3)
cv2.imshow("Detected Lines", edges)
cv2.waitKey(0)

# # Морфологические операции: сначала открытие, затем закрытие
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow("Detected Lines", morph)
cv2.waitKey(0)

# Находим линии с помощью HoughLinesP
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

vertical_lines = []
horizontal_lines = []

if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        # Вычисляем угол линии
        angle = np.degrees(np.arctan2((y2-y1), (x2-x1)))
        # Вертикальные линии: угол около ±90 градусов
        if abs(angle) > 75:
            vertical_lines.append((x1,y1,x2,y2))
        # Горизонтальные линии: угол около 0 градусов
        elif abs(angle) < 15:
            horizontal_lines.append((x1,y1,x2,y2))

# Для отладки можно отобразить найденные линии
lines_img = img.copy()
for x1,y1,x2,y2 in vertical_lines:
    cv2.line(lines_img, (x1,y1), (x2,y2), (0,0,255), 2)
for x1,y1,x2,y2 in horizontal_lines:
    cv2.line(lines_img, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.imshow("Detected Lines", lines_img)
cv2.waitKey(0)

# Кластеризуем вертикальные линии по x и горизонтальные по y
v_clusters = cluster_lines(vertical_lines, axis=0, thresh=20)
h_clusters = cluster_lines(horizontal_lines, axis=1, thresh=20)

print("Вертикальные линии (x):", v_clusters)
print("Горизонтальные линии (y):", h_clusters)

# Если сетка должна быть 3х3, то у нас должно получиться 3 вертикальные и 3 горизонтальные линии (или 4 если учитывать границы)
# Здесь мы выбираем 3 наиболее значимых (можно доработать по необходимости)
if len(v_clusters) < 3 or len(h_clusters) < 3:
    print("Не найдено достаточное количество линий для формирования сетки!")
    exit(1)

# Если найдено больше линий, выберем равномерно
v_clusters = sorted(v_clusters)[:3]
h_clusters = sorted(h_clusters)[:3]

# Формируем точки-пересечения из кластеров
grid_points = []
for x in v_clusters:
    for y in h_clusters:
        grid_points.append((int(x), int(y)))
grid_points = np.array(grid_points).reshape((3,3,2))
# grid_points теперь имеет форму 3x3, как и ранее заданная вручную

# Определяем 4 квадрата как области между соседними точками
# Порядок вершин каждого квадрата: [топ-лево, топ-право, бот-право, бот-лево]
square1 = np.array([grid_points[0,0], grid_points[0,1], grid_points[1,1], grid_points[1,0]])
square2 = np.array([grid_points[0,1], grid_points[0,2], grid_points[1,2], grid_points[1,1]])
square3 = np.array([grid_points[1,0], grid_points[1,1], grid_points[2,1], grid_points[2,0]])
square4 = np.array([grid_points[1,1], grid_points[1,2], grid_points[2,2], grid_points[2,1]])
squares = [square1, square2, square3, square4]

# Для каждого квадрата назначим свой цвет
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
result_img = img.copy()

# Отрисовываем найденную сетку (точки)
for row in grid_points:
    for pt in row:
        cv2.circle(result_img, tuple(pt), 5, (255,255,0), -1)

# Отрисовываем квадраты
for idx, sq in enumerate(squares):
    cv2.polylines(result_img, [sq], isClosed=True, color=colors[idx], thickness=3)
    M = cv2.moments(sq)
    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        cv2.putText(result_img, f"Square {idx+1}", (cx-20,cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[idx], 2)

cv2.imshow("Detected Grid Squares", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
