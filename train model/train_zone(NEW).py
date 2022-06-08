import os
from pickle import load, dump
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
import string
import numpy as np


def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # print(model.summary())
    features = dict()
    for name in os.listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        # print('>', name)
    return features


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split('.')
        if len(tokens) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_desc = ''.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
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


def to_vocabulary(descriptions):
    all_desc = set()
    for value in descriptions.values():
        [all_desc.update(d.split()) for d in value]
    return all_desc


def save_descriptions(descriptions, filename):
    lines = []
    for image_id, image_desc in descriptions.items():
        for desc in image_desc:
            lines.append(image_id + '\t' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_set(filename):
    doc = load_doc(filename)
    dataset = []
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split('\t')
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions


def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def dict_to_list(descriptions):
    all_desc = []
    for values in descriptions.values():
        [all_desc.append(x) for x in values]
    return all_desc


def create_tokenizer(descriptions):
    lines = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    x1, x2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            input_seq, output_seq = seq[:i], seq[i]
            input_seq = pad_sequences([input_seq], maxlen=max_length)[0]
            output_seq = to_categorical([output_seq], num_classes=vocab_size)[0]
            x1.append(photo)
            x2.append(input_seq)
            y.append(output_seq)
    return np.array(x1), np.array(x2), np.array(y)


def max_length(descriptions):
    lines = dict_to_list(descriptions)
    return max(len(d.split()) for d in lines)


dataset_img = "DATASETS\\Flickr8k_Dataset\\Flicker8k_Dataset(TEST)"
features = extract_features(dataset_img)
# print('Extracted features: ', len(features))
dump(features, open('features(NEW).p', 'wb'))

dataset_txt = 'DATASETS\\Flickr8k_text\\'
txt_token = dataset_txt + 'Flickr8k.token(TEST).txt'
doc = load_doc(txt_token)
# print(doc.split('\n'))

descriptions = load_descriptions(doc)
# print('Loaded: ', len(descriptions))
# print(descriptions)

descriptions = clean_descriptions(descriptions)
# print(descriptions)

vocab = to_vocabulary(descriptions)
# print(vocab)
# print('Vocabulary size:', len(vocab))

save_descriptions(descriptions, 'descriptions(NEW).txt')

trainImages = dataset_txt + '/Flickr_8k.trainImages(TEST).txt'
train = load_set(trainImages)
# print('Training images loaded:', len(train))
# print(train)

train_descriptions = load_clean_descriptions('descriptions(NEW).txt', train)
# print("Training descriptions loaded:", len(train_descriptions))
# print(train_descriptions)

train_features = load_photo_features('features(NEW).p', train)
# print('Features loaded:', len(train_features.keys()))
# for i, feature in enumerate(features.values()):
# print(f'@{i} - {feature}')

tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer(NEW).p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
# print('Vocabulary size:', vocab_size)

max_length = max_length(train_descriptions)
# print('Maximum sentence length:', max_length)


def define_model(vocab_size, max_length):
    # Извлечение особенностей
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # Процессор последовательностей
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # Декодер
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # Связывание всего в единое
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Подведение итогов
    # print(model.summary())
    # plot_model(model, to_file='models(NEW)/model.png', show_shapes=True)
    return model


def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while 1:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield [in_img, in_seq], out_word


generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
inputs, outputs = next(generator)
# print(inputs[0].shape)
# print(inputs[1].shape)
# print(outputs.shape)


model = define_model(vocab_size, max_length)
epochs = 75
steps = len(train_descriptions)
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    model.save('models(NEW)/model_' + str(i + 1) + '.h5')