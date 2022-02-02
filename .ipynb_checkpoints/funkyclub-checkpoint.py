import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load

from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

from tqdm import notebook
notebook.tqdm().pandas()


def read_document(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


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


def unique_words_vocab(descriptions):
    unique_words = set()

    for key in descriptions.keys():
        [unique_words.update(desc.split()) for desc in descriptions[key]]

    return unique_words


def save_descriptions(descriptions, filename):
    lines = []
    for img_name, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(img_name + '\t' + desc)
        text = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(text)
        file.close()


def extract_features(directory):
    model = Xception(include_top=False, pooling='Avg')
    features = {}
    for img in notebook.tqdm(os.listdir(directory)):
        filename = directory + '/' + img
        image = (Image.open(filename)).resize(299, 299)
        image = np.expand_dims(image, axis=0)
        image = (image / 127.5) - 1

        feature = model.predict(image)
        features[img] = feature

    return features


dataset_text = "D:\\DATASETS\\Flickr8k_Text"
dataset_img = "D:\\DATASETS\\Flickr8k_Dataset"

filename = dataset_text + "\\" + "Flickr8k.token.txt"

descriptions = all_img_names(filename)
clean_desc = text_cleaning(descriptions)
unique_words = unique_words_vocab(descriptions)

save_descriptions(clean_desc, "descriptions.txt")

features = extract_features(dataset_img)
dump(features, open('features.p', 'wb'))

#[print('[' + thing + '],', ':', '[' + all_img_names(filename)[thing] + '],') for thing in all_img_names(filename)]
#print(unique_words)



