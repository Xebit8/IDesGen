import unittest
from PIL import Image
from tensorflow.keras.applications.xception import Xception
import numpy as np


def extract_features(filename, model): # Функция извлечения особенностей изображения
    try: # Проверка на открытие файла по правильному пути
        image = Image.open(filename)
    except:
        print("Ошибка! Возможно, что был выбран неправильный путь.")
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4: # Проверка файла на соответствие формату изображения, а не мультимедиа или «.GIF»
        image = image[..., :3]
    # Подготовка изображения для удобного извлечения особенностей
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image) # Извлечение особенностей
    return feature


img_path = "gift.jpg"
gif_path = "little_friend.gif"
xception_model = Xception(include_top=False, pooling='avg')

# Класс для проведения испытаний – позитивного и негативного тест-кейсов
class TestProgram(unittest.TestCase):

    # Использование для проверки изображения
    def test_positive(self):
        extract_features(img_path, xception_model)

    # Использование для проверки файла формата «.GIF»
    def test_negative(self):
        extract_features(gif_path, xception_model)


if __name__ == '__main__':
    unittest.main()

