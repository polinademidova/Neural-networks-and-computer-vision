import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random

# Загружаем готовую модель
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Загружаем тестовые данные
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255.0  # нормализация

# Названия классов
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_image(index):
    img = x_test[index]
    img_expanded = img.reshape(1, 28, 28)  # батч из 1 картинки
    predictions = model.predict(img_expanded)
    predicted_class = predictions[0].argmax()

    # Оцениваем точность один раз (лучше вынести из функции)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Визуализация
    plt.imshow(img, cmap='gray')
    plt.title(f'Предсказано: {class_names[predicted_class]} ({test_accuracy:.2f})\n'
              f'Истинный класс: {class_names[y_test[index]]}')
    plt.axis('off')
    plt.show()

random_index = random.randint(0, len(x_test) - 1)
predict_image(random_index)
