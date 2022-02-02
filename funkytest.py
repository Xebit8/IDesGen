# from pickle import load
# from typing import List
# import tensorflow
from PIL import Image
import numpy as np

# x = ' '
# y = ' '
# z = ".,-!?/;:'"'"'"()*&^%$#@~`[]{}\|№/<>"
# str = "ftp://dotc.ru/?"
# table = str.maketrans(x, y, z)
# print(table)
# print(str.translate(table))


# text = "Img1#1\tToday is a good day .\n" \
#        "Img1#2\tToday is the better day .\n" \
#        "Img1#3\tToday is a great day .\n" \
#        "Img2#1\tToday is a bad day .\n" \
#        "Img2#2\tToday is the worse day .\n" \
#        "Img3#1\tToday is a strange day .\n"
# dicty = {}
# textlines = text.split('\n')
# for textline in textlines[:-1]:
#     img, caption = textline.split('\t')
#     if img[:-2] not in dicty:
#         dicty[img[:-2]] = [caption]
#     else:
#         dicty[img[:-2]].append(caption)
#
# [print(num_part, ':', cap_part.replace('-', ' ')) for num_part, cap_part in enumerate([cap for img, cap in dicty.items()])]
# for img, cap in dicty.items():
#     for num_part, cap_part in enumerate(cap):
#         #cap_part.replace("-", " ")
#         desc = cap_part.split()
#         # print(desc)
#         desc = [word.lower() for word in desc]
#         print(desc)


# filename = "Flickr8k_text\\Flickr_8k.testImages.txt"
# descriptions = {}
#
# file = open(filename, 'r')
# text = file.read()
# file.close
#
# for line in text.split('\n'):
#     words = line.split()
#     if len(words)<1:
#         continue
#
#     image, image_caption = words[0], words[:1]
#     # print(f'words = {words},\nimage = {image},\nimage_caption = {image_caption}')
#
#     if image in text:
#         if image not in descriptions:
#             descriptions[image] = []
#         desc = '<start> ' + " ".join(image_caption) + ' <end>'
#         descriptions[image].append(desc)


# def read_document(filename):
#     file = open(filename, 'r')
#     text = file.read(5000)
#     file.close()
#     return text
#
#
# def read_photos(filename):
#     filename = read_document(filename)
#     photo_lines = filename.split('\n')[:-1]
#     return photo_lines
#
#
# def load_features(photos):
#     all_features = load(open('features.p', 'rb'))
#     features = {all_features[k] for k in photos}
#     return features
#
#
# filename = dataset_img = "Flickr8k_Dataset\\Flickr8k_Dataset(test_version)"
#
# train_images = read_photos(filename)
# load_features()
#
# print()

# [print(f'key = {key}, value = {value}') for key, value in descriptions.items()]

# primes: List[int] = [1, 2, 3, 4, 5]
#
# [print(num) for num in primes]


# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# filename = "Flickr8k_Dataset\\Flickr8k_Dataset(test_version)\\667626_18933d713e.jpg"
#
# image = Image.open(filename)
# image = image.resize((299, 299))
# image = np.expand_dims(image, axis=0)
# image = image - 127.5
# image = image - 1
# print(image)

# filename = 'somefilesnamedpath'
# a =
# b = 4




