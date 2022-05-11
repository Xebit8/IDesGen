import PySimpleGUI as sg
import shutil
from googletrans import Translator
import os
from PIL import Image
import time
import urllib.request
import clipboard as cp
from caption_generator import max_length, tokenizer, model, CaptionGenerator
from database import DatabaseInput, DatabaseOutput


# Проверяем, подключен ли компьютер к сети
def connect(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


# Перевод изображений в формат PNG с одним размером. Изображения такого формата сохраняются в папку temp
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


# Окно с инструкцией
def instruction():
    def border(elem):
        return sg.Frame('', [[elem]], background_color='#3721AF')

    instruction_text = '''      Для начала работы генератора подписей необходимо выбрать изображение. Для этого нужно 
нажать на кнопку "Выбрать изображение". После этого появится кнопка "Начать", которая
и запустит процесс генерации подписи. При первичном запуске, процесс может занять больше
времени. В результате в прямоугольнике появится подпись к изображению. Кнопка "Скопировать"
служит для копирования подписи. 
        После получения первого результата, вам будет доступна кнопка "История". Перейдя 
по этой кнопке в новое окно, вы получите возможность просмотра всех результатов, которые
выдавала программа по вашим изображениям. Переход по результатам возможен путём исполь-
зования стрелочек (< >). Вы также можете нажать на кнопку "Очистить историю", чтобы 
очистить историю результатов. Кнопка "Назад" вернёт вас обратно в главное окно.'''
    layout = [[border(sg.Button('Назад', size=(20, 2), mouseover_colors='#3721AF')), border(sg.Text(text=instruction_text, background_color='#FFFFFF', text_color='#000000'))]]
    window = sg.Window('Инструкция',
                       layout,
                       background_color='#FFFFFF',
                       button_color=('#3721AF', 'white'),
                       font='SourceSansPro',
                       resizable=False)
    return window


# Окно с историей
def history():
    def border(elem):
        return sg.Frame('', [[elem]], background_color='#3721AF')

    left_side = [[border(sg.Button('<', size=(2, 2), key='left_btn', mouseover_colors='#3721AF'))]]
    right_side = [[border(sg.Button('>', size=(2, 2), key='right_btn', mouseover_colors='#3721AF'))]]
    center = [[border(sg.Image(background_color='#3721AF', size=(520, 356), key='img'))],
              [sg.Text("", size=(47, 2), background_color='#3721AF', justification='c', key='caption')],
              [sg.Button('Скопировать', key='copy_history', button_color='#3721AF')],
              [border(sg.Button('Назад', size=(20, 2)))]]
    layout = [[sg.Column([[border(sg.Button('Очистить историю', key='clear', button_color='#3721AF', mouseover_colors=('#3721AF', 'white')))]], justification='r', background_color='white')],
              [sg.Column(left_side, element_justification='c', background_color='#FFFFFF'),
              sg.Column(center, element_justification='c', background_color='#FFFFFF'),
              sg.Column(right_side, element_justification='c', background_color='#FFFFFF')]]
    window = sg.Window('История',
                       layout,
                       background_color='#FFFFFF',
                       button_color=('#3721AF', 'white'),
                       font='SourceSansPro',
                       resizable=False)
    return window


# Главное окно программы
def main_window():

    def border(elem):
        enabled = sg.Frame('', [[elem]], background_color='#3721AF')
        return enabled

    # Сторона главного окна с кнопками
    input_side = [[border(sg.Button('Инструкция', size=(20, 2), mouseover_colors='#3721AF'))],
                  [border(sg.Button('Выбрать изображение',
                                    button_type=2,
                                    key='Browse',
                                    target='file_browsed',
                                    size=(20, 2),
                                    mouseover_colors='#3721AF'))],
                  [sg.Frame('',
                            [[sg.Button('Начать', size=(20, 2), disabled=True, key='start', mouseover_colors='#3721AF')]],
                            background_color='#3721AF',
                            key='disabled_start',
                            visible=False)],
                  [sg.Frame('',
                            [[sg.Button('История', size=(20, 2), disabled=True, key='history', mouseover_colors='#3721AF')]],
                            background_color='#3721AF',
                            key='disabled_history',
                            visible=False)]]

    # Сторона главного окна с изображением и его подписью
    output_side = [[border(sg.Image(background_color='#3721AF', key='img', size=(520, 356)))],
                   [sg.Text("", size=(47, 2), key='caption', background_color='#3721AF', justification='c')],
                   [sg.Input("", size=(47, 2), enable_events=True, key='file_browsed', visible=False)],
                   [sg.Button('Скопировать', key='copy_main', button_color='#3721AF')]]

    # Главное окно целиком
    layout = [[sg.Column(input_side, element_justification='c', background_color='#FFFFFF'), sg.Column(output_side, element_justification='c', background_color='#FFFFFF')]]

    # Создание окна приложения и его настройка
    window = sg.Window("«IDesGen» - генератор подписей к изображениям",
                       layout,
                       background_color='#FFFFFF',
                       button_color=('#3721AF', 'white'),
                       font='SourceSansPro',
                       resizable=False)

    return window


# Главное окно программы
def interface(max_length, tokenizer, model):
    window = main_window()

    cg = CaptionGenerator()
    dbi = DatabaseInput()
    dbo = DatabaseOutput()
    tokenizer = tokenizer
    max_length = max_length
    model = model
    ru_caption = ""
    caps = []
    cap = ''
    imgs = []
    image_id = 0

    while True:
        event, values = window.read(timeout=1)

        # print(f'event = {event}')
        # print(f'values = {values}')

        # Проявление кнопки "История" при наличии данных в БД
        if dbi.tables_check() and 'history' in window.AllKeysDict:
            window['history'].update(disabled=False)
            window['disabled_history'].update(visible=True)
        event, values = window.read(timeout=100000)

        # При нажатии на крестик в правом-верхнем углу программа удалит папку Temp и закроет окно приложения
        if event == sg.WIN_CLOSED :
            shutil.rmtree('temp', ignore_errors=True)
            break

        # Если нажата кнопка "Выбрать изображение"
        if event == 'file_browsed':
            window['caption'].update("")
            # Если пользователь ничего не выбрал, то кнопка "Начать" не появится
            if window['file_browsed'].get() != "":
                window['start'].update(disabled=False)
                window['disabled_start'].update(visible=True)
            img_path = values['file_browsed']
            print(f'img_path: {img_path}')
            try:
                picture = convertToPNG(img_path)
            except AttributeError:
                continue
            print(f'picture: {picture}')
            window['img'].update(picture)

        # При нажатии на кнопку выбранное изображение появится в рамке
        if event == 'start':
            # Генерация подписи изображения
            photo = cg.extract_features(values['file_browsed'])
            description = cg.generate_desc(model, tokenizer, photo, max_length)

            # Обработка подписи (чистка, перевод)
            caption = clean_caption(description)
            if connect():
                caption = translate_caption_lang(caption)
                window['caption'].update(caption)
            else:
                window['caption'].update(caption)
            dbi.fill_tables(img_path, picture, caption)
            window['history'].update(disabled=False)

        # Переход по объектам БД налево
        if event == 'left_btn':
            # print(image_id)
            # print(caps)
            # print(id_list)
            if 0 <= image_id - 1:
                window['img'].update(imgs[image_id - 1])
                cap = caps[image_id - 1]
                window['caption'].update(caps[image_id - 1])
                image_id -= 1
                window.refresh()

        # Переход по объектам БД направо
        if event == 'right_btn':
            # print(image_id)
            # print(caps)
            # print(id_list)
            if len(caps) - 1 >= image_id + 1:
                window['img'].update(imgs[image_id + 1])
                cap = caps[image_id + 1]
                window['caption'].update(caps[image_id + 1])
                image_id += 1
                window.refresh()

        # Копирование подписи на главном экране
        if event == 'copy_main':
            try:
                cp.copy(caption)
            except UnboundLocalError:
                pass
        # Копирование подписи на экране истории
        if event == 'copy_history':
            cp.copy(cap)
        # Переход в окно с инструкцией
        if event == 'Инструкция':
            window.close()
            window = instruction()
        # Переход в главное окно
        if event == 'Назад':
            window.close()
            window = main_window()
        # Очистка базы данных
        if event == 'clear':
            window.close()
            dbo.drop_tables()
            window = main_window()
        # Переход в окно истории
        if event == 'history':
            window.close()
            window = history()
            window.read(timeout=1)
            imgs = dbo.image_array_from_db()
            # print(type(imgs))
            caps = dbo.image_caption_from_db()
            # print(caps, type(caps))
            id_list = dbo.image_id_from_db()
            print(id_list, type(id_list))
            image_id = id_list[-1]
            window['img'].update(imgs[-1])
            window['caption'].update(caps[-1])
            cap = caps[-1]


if __name__ == "__main__":
    interface(max_length, tokenizer, model)