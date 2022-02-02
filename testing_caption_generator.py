import numpy as np
from PIL import Image
from pickle import load
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception
import PySimpleGUI as sg
# from io import BytesIO
import shutil
import clipboard as cp
from googletrans import Translator


# Перевод изображений в формат PNG с одним размером. Изображения такого формата сохраняются в папку Temp
def convertToPNG(img_path):
    if os.path.exists('temp') is False:
        os.makedirs('temp')
    image = Image.open(img_path)
    image = image.resize((520, 356))
    picture = 'temp\\' + str(len(os.listdir('temp'))+1) + '.png'
    image.save(picture)
    return picture
    # with BytesIO() as f:
    #     image.save(f, format='PNG')
    # f.seek(0)
    # image_png = Image.open(f)
    # return image_png


# Очистка подписей от меток <start>, <end>
def clean_caption(description):
    caption = description.split(' ')
    del caption[0]
    del caption[-1]
    clean_caption = ' '.join(caption)
    return clean_caption


# Перевод подписей на русский язык
def translate_caption_lang(caption):
    translator = Translator()
    translation = translator.translate(caption, dest='ru')
    caption = translation.text
    return caption


# Извлечение особенностей из изображений
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("Ошибка! Возможно, что был выбран неправильный путь.")
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature


# Сбор словаря уникальных слов
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Сбор подписи, генератор подписей
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        pred = model.predict([photo, seq], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


# Сторона главного окна с кнопками
input_side = [[sg.Button('Инструкция', size=(20, 2))],
              [sg.FileBrowse('Выбрать изображение', key='-INPUT-', target='input-filename', size=(20, 2))],
              [sg.Button('Начать', size=(20, 2))]]

# Сторона главного окна с изображением и его подписью
output_side = [[sg.Image(background_color='#887CBC', key='img', size=(520, 356))],
               [sg.Text("", size=(47, 2), key='output-filename', background_color='#49426B', justification='c')],
               [sg.Text("", size=(47, 2), key='input-filename', visible=False, background_color='#49426B', justification='c')],
               [sg.Button('Скопировать')]]

# Главное окно целиком
layout = [[sg.Column(input_side, element_justification='c', background_color='#AFA4E0'), sg.Column(output_side, element_justification='c', background_color='#AFA4E0')]]

# Создание окна приложения и его настройка
window = sg.Window("«IDesGen» - генератор подписей к изображениям",
                   layout,
                   use_custom_titlebar=True,
                   titlebar_background_color='#2E2844',
                   titlebar_font='SourceSansPro 10',
                   background_color='#AFA4E0',
                   button_color='#49426B',
                   font='SourceSansPro',
                   resizable=False)

img_path = ""

max_length = 32

while True:
    event, values = window.read()

    # При нажатии на крестик в правом-верхнем углу программа удалит папку Temp и закроет окно приложения
    if event == sg.WIN_CLOSED:
        shutil.rmtree('temp', ignore_errors=True)
        break

    # При нажатии на кнопку выбранное изображение появится в рамке
    if event == 'Начать':
        img_path = values['-INPUT-']
        print(f'img_path: {img_path}')
        picture = convertToPNG(img_path)
        print(f'picture: {picture}')
        window['img'].update(picture)

    #  Генерация подписи изображения
        tokenizer = load(open("tokenizer.p", 'rb'))
        model = load_model('models/model_9.h5')
        xception_model = Xception(include_top=False, pooling='avg')
        photo = extract_features(img_path, xception_model)
        description = generate_desc(model, tokenizer, photo, max_length)

    # Обработка подписи (чистка, перевод)
        caption = clean_caption(description)
        ru_caption = translate_caption_lang(caption)

        window['output-filename'].update(ru_caption)
    # Копирование текстового содержимого подписи  при нажатии на кнопку
    if event == 'Копировать':
        cp.copy(ru_caption)

