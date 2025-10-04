from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os

DATASET_PATH = "C:/PythonProjects/neuralnetworks/pythonProject2/dataset/"
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f'Ошибка: Файл не найден по пути: {image_path}')
        return

        # Проверка на повреждение
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f'Ошибка: Поврежденное изображение - {image_path}')
        return

        # Загружаем модель
    model = tf.keras.models.load_model("image_classifier.h5")

    # Читаем картинку через OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f'Ошибка: Не удалось прочитать изображение - {image_path}')
        return

        # Подготовка изображения
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    # Предсказание
    prediction = model.predict(img)
    class_names = sorted(os.listdir(DATASET_PATH))  # сортируем для стабильности

    predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]

    print(f'Модель определила: {predicted_class}')

    #Визуализируем результат
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'Модель определила: {predicted_class}')
    plt.axis('off')
    plt.show()

# Пример вызова
predict_image(os.path.join(DATASET_PATH, "cats/cat1.jpg"))
