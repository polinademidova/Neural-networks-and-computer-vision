import cv2
from matplotlib import pyplot as plt
import os

# Пути к файлам (проверь, что они реально существуют!)
image_path = r"C:\PythonProjects\neuralnetworks\pythonProject2\haarcascade\people3.jpg"
cascade_path = r"C:\PythonProjects\neuralnetworks\pythonProject2\haarcascade\haarcascade_frontalface_alt2.xml"

# Загружаем изображение
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Не удалось открыть картинку: {image_path}")

# Преобразование в градации серого и RGB
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Загружаем классификатор Хаара
face_data = cv2.CascadeClassifier(cascade_path)
if face_data.empty():
    raise FileNotFoundError(f"Не удалось открыть XML файл каскада: {cascade_path}")

# Поиск лиц
faces = face_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
print("Найденные лица:", faces)

# Рисуем кружки вокруг лиц
for (x, y, width, height) in faces:
    cv2.circle(img_rgb, (x + width // 2, y + height // 2), width // 2, (0, 255, 0), 5)

# Отображаем результат
plt.imshow(img_rgb)
plt.axis("off")
plt.show()