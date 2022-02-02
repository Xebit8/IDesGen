import string
import numpy as np
from PIL import Image
import os
# from database import tokenizer_from_db, captions_from_db, features_from_db, tokenizer_to_db, captions_to_db, features_to_db
from pickle import dump, load

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout

from tqdm import notebook
notebook.tqdm().pandas()


# Открытие текстового документа и вывод содержимого
def read_document(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# Сбор словаря названий изображений и подписей изображений из файла
def all_img_names(filename):
    file = read_document(filename)
    desc_lines = file.split('\n')
    descriptions = {}
    for desc_line in desc_lines[:-1]:
        img, caption = desc_line.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions


# Очистка подписей
def text_cleaning(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for img, cap in descriptions.items():
        for num_part, caption_part in enumerate(cap):

            caption_part.replace('-', ' ')
            desc = caption_part.split()

            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if(len(word) > 1)]
            desc = [word for word in desc if(word.isalpha())]

            caption_part = ' '.join(desc)
            descriptions[img][num_part] = caption_part

    return descriptions


# Создание словаря уникальных слов
def unique_words_vocab(descriptions):
    unique_words = set()

    for key in descriptions.keys():
        [unique_words.update(desc.split()) for desc in descriptions[key]]

    return unique_words


# Сохранение очищенных подписей в текстовый файл
def save_descriptions(descriptions, filename):
    lines = []
    for img_name, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(img_name + '\t' + desc)
    text = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(text)
    file.close()


# Пути к тренировочным данным
dataset_text = "Flickr8k_Text"
dataset_img = "Flickr8k_Dataset\\Flickr8k_Dataset(test_version)"

token_filename = dataset_text + "\\" + "Flickr8k.token(test_version).txt"

descriptions = all_img_names(token_filename)
clean_desc = text_cleaning(descriptions)
unique_words = unique_words_vocab(descriptions)

save_descriptions(clean_desc, "descriptions.txt")


# Извлечение особенностей изображений
def extract_features(directory):
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img in notebook.tqdm(os.listdir(directory)):
        filename = directory + '/' + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0

        feature = model.predict(image)
        features[img] = feature

    return features


features = extract_features(dataset_img)
dump(features, open('features.p', 'wb'))
# features_to_db(features)


# Чтение файла с фото
def read_photos(filename):
    filename = read_document(filename)
    photo_lines = filename.split('\n')[:-1]
    return photo_lines


# Обработка словаря подписей с метками
def load_clean_desc(filename, photos):
    file = read_document(filename)
    descriptions = {}

    for line in file.split('\n')[:-1]:
        # print("file len =", len(file))
        # print("file:",file)
        # print("line:",line)
        words = line.split('\t')
        #print("words:",words)
        if len(words) < 1:
            continue

        image, image_caption = words[0], words[1:]
        #print(f'image: {image}, image_caption: {image_caption}')
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions


# Чтение особенностей
def load_features(photos):
    all_features = load(open('features.p', 'rb'))
    # all_features = features_from_db()
    features = {n: all_features[n] for n in photos}
    return features


train_img_filename = dataset_text + "\\" + "Flickr_8k.trainImages(test_version).txt"

train_img = read_photos(train_img_filename)
clean_train_desc = load_clean_desc("descriptions.txt", train_img)
#print("clean_train_desc =", clean_train_desc)
train_features = load_features(train_img)
#print("train_features =", train_features)


# Перевод словаря в тип списка
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(value) for value in descriptions[key]]
    return all_desc


# Создание словаря лексем
def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


tokenizer = create_tokenizer(clean_train_desc)
# print("tokenizer =", tokenizer.get_config())
# tokenizer_to_db(tokenizer)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
#print("vocab_size =", vocab_size)


# Определение максимального размера
def get_max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)


max_length = get_max_length(descriptions)
#print("max_length =", max_length)

#print(f'clean_train_desc: {clean_train_desc}, features: {features}, tokenizer: {tokenizer}, max_length: {max_length}')


# Создание генератора данных на основе собранных свойств
def data_generator(descriptions, features, tokenizer, max_length):
    while True:
        for key, description_list in descriptions.items():
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield [[input_image, input_sequence], output_word]


#print(f'data_generator: {data_generator(clean_train_desc, features, tokenizer, max_length)}')


# Создание последовательностей для генератора (а дальше - для модели)
def create_sequences(tokenizer, max_length, description_list, feature):
    X1, X2, Y = list(), list(), list()
    for desc in description_list:
        seq = Tokenizer.texts_to_sequences(tokenizer, [desc])[0]
        for i in range(1, len(seq)):
            input_seq, output_seq = seq[:i], seq[i]
            input_seq = pad_sequences([input_seq], maxlen=max_length)[0]
            output_seq = to_categorical([output_seq], num_classes=vocab_size)[0]

            X1.append(feature)
            X2.append(input_seq)
            Y.append(output_seq)

    return np.array(X1), np.array(X2), np.array(Y)


[[a, b], c] = next(data_generator(clean_train_desc, train_features, tokenizer, max_length))
print(f'a.shape = {a.shape}, b.shape = {b.shape}, c.shape = {c.shape}')


# Определение структуры модели обучения
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    ftr1 = Dropout(0.5)(inputs1)
    ftr2 = Dense(256, activation='relu')(ftr1)

    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)
    seq3 = LSTM(256)(seq2)

    decoder1 = add([ftr2, seq3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    print(model.summary())
    plot_model(model, show_shapes=True, to_file='model.png')

    return model


# Проверка основных данных
print(f'Dataset: {len(train_img)}')
print(f'Descriptions: {len(clean_train_desc)}')
print(f'Photos: {len(train_features)}')
print(f'Vocabulary size: {vocab_size}')
print(f'Description max length: {max_length}')

# Тренировка модели
model = define_model(vocab_size, max_length)
epochs = 10
steps = len(clean_train_desc)
os.mkdir('models')
for i in range(epochs):
    generator = data_generator(clean_train_desc, train_features, tokenizer, max_length)
    print('steps_per_epoch =', steps)
    model.fit(x=generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("models/model" + str(i) + ".h5")
