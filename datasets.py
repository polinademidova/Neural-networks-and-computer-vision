from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
import os

# Путь к датасету
DATASET_PATH = 'dataset/'

# Определяем количество классов
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

# Генератор изображений с нормализацией + аугментация для устойчивости модели
train_datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=20,         # случайные повороты
    width_shift_range=0.1,     # сдвиг по ширине
    height_shift_range=0.1,    # сдвиг по высоте
    shear_range=0.1,           # сдвиги
    zoom_range=0.1,            # масштабирование
    horizontal_flip=True,      # отзеркаливание
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

# Загружаем тренировочные данные
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='training'
)

# Загружаем валидационные данные
val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='validation'
)

model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# Компиляция
loss_function = 'binary_crossentropy' if class_mode == 'binary' else 'categorical_crossentropy'
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=30, verbose=1)

# Оценка модели
test_loss, test_accuracy = model.evaluate(val_data)
print(f'Точность модели на валидационных данных: {test_accuracy:.2f}')

# Сохраняем модель
model.save('image_classifier.h5')