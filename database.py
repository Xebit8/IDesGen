import sqlite3
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


# Подключение базы данных
conn = sqlite3.connect('train_data_database.db')
conn.execute("PRAGMA foreign_keys = 1")
cursor = conn.cursor()


class DataBaseInput:

    def __init__(self, tokenizer=None, clean_train_desc=None, features=None):
        self.tokenizer = tokenizer
        self.captions = clean_train_desc
        self.features = features
        self.tokenizer_table = self.tokenizer_to_db()
        self.captions_table = self.captions_to_db()
        self.features_table = self.features_to_db()

    # Создание таблицы-словаря лексемов (токенизатора)
    def tokenizer_to_db(self):
        print(f'tokenizer = {self.tokenizer}')
        if self.tokenizer is None:
            pass
        else:
            cursor.execute('''CREATE TABLE IF NOT EXISTS Tokenizer
            (word_ID INTEGER PRIMARY KEY,
            num_words NULL, 
            filters TEXT NOT NULL,
            lower TEXT NOT NULL, 
            split TEXT, 
            char_level TEXT NOT NULL,
            oov_token NULL, 
            document_count INTEGER NOT NULL,
            word TEXT NOT NULL,
            repetitions INTEGER NOT NULL)''')

            tokenizer_table = []
            i = 0

            # Обработка данных и перевод в нужный формат
            token_config = self.tokenizer.get_config()
            words = self.tokenizer.get_config()['word_counts']
            words = words[1:len(words)-1]
            words = dict(item.split(': ') for item in words.replace('"', '').split(', '))
            for key, value in words.items():
                words[key] = int(value)

            for word, num in words.items():
                tokenizer_table.append((i, token_config['num_words'], token_config['filters'],
                                        token_config['lower'], token_config['split'], token_config['char_level'],
                                        token_config['oov_token'], token_config['document_count'], word, num))
                i += 1

            # [print(tokenizer_table[i]) for i in range(len(tokenizer_table))]
            return tokenizer_table

    # Создание таблицы подписей к изображениям
    def captions_to_db(self):
        print(f'captions = {self.captions}')
        if self.captions is None:
            pass
        else:
            cursor.execute('''CREATE TABLE IF NOT EXISTS Captions
            (image_ID INTEGER PRIMARY KEY,
            image_name TEXT NOT NULL,
            captions_list TEXT NOT NULL)''')

            captions_table = []
            i = 0

            # Обработка данных и перевод в нужный формат
            for image_name, caption_list in self.captions.items():
                caption_list = ", ".join(caption_list)
                captions_table.append((i, image_name, caption_list))
                i += 1

            # [print(captions_table[i]) for i in range(len(captions_table))]
            return captions_table

    # Создание таблицы особенностей изображений
    def features_to_db(self):
        print(f'features = {self.features}')
        if self.features is None:
            pass
        else:
            cursor.execute('''CREATE TABLE IF NOT EXISTS Features
            (image_ID INTEGER PRIMARY KEY,
            image_name TEXT NOT NULL,
            feature TEXT NOT NULL,
            FOREIGN KEY (image_ID) REFERENCES Captions (image_ID))''')

            features_table = []
            i = 0

            # Обработка данных и перевод в нужный формат
            for img_name, feature in self.features.items():
                features_table.append((i, img_name, feature.tostring()))
                i += 1

            # [print(features_table[i]) for i in range(len(features_table))]
            return features_table

    # Заполнение данными таблицы
    def fill_tables(self):
        try:
            cursor.executemany("INSERT INTO Tokenizer VALUES (?,?,?,?,?,?,?,?,?,?)", self.tokenizer_table)
            cursor.executemany("INSERT INTO Captions VALUES (?,?,?)", self.captions_table)
            cursor.executemany("INSERT INTO Features VALUES (?,?,?)", self.features_table)
        except sqlite3.IntegrityError:
            pass

        conn.commit()


class DataBaseOutput:
    def __init__(self):
        self.retrieved_tokenizer = self.tokenizer_from_db()
        self.retrieved_captions = self.captions_from_db()
        self.retrieved_features = self.features_from_db()

    # Просмотр и сравнение данных
    def view_data(self):
        view_tokenizer = "SELECT * FROM Tokenizer"
        cursor.execute(view_tokenizer)
        print('Токен из Базы Данных:')
        for row in cursor.fetchall():
            print(row)

        view_captions = "SELECT * FROM Captions"
        cursor.execute(view_captions)
        print('Подписи из Базы Данных:', cursor.fetchall())

        view_features = "SELECT * FROM Features"
        cursor.execute(view_features)
        print('Особенности из Базы Данных:', cursor.fetchall())

        # print('\n\n\n')
        # print('Токен исходный:', self.tokenizer.get_config())
        # print('Подписи исходные:', self.captions)
        # print('Особенности исходные:', self.features)

    # Вывод информации из базы данных в нужном формате
    def tokenizer_from_db(self):
        query = "SELECT num_words, filters, lower, split, char_level, oov_token, document_count from Tokenizer"
        cursor.execute(query)
        rows = cursor.fetchall()
        retrieved_tokenizer = {}
        for row in rows:
            num_words = row[0]
            filters = row[1]
            lower = row[2]
            split = row[3]
            char_level = row[4]
            oov_token = row[5]
            document_count = row[6]
            retrieved_tokenizer = dict([('num_words', num_words), ('filters', filters), ('lower', lower), ('split', split), ('char_level', char_level), ('oov_token', oov_token), ('document_count', document_count)])
        query = "SELECT word, repetitions FROM Tokenizer"
        cursor.execute(query)
        rows = cursor.fetchall()
        word_dict = []
        for row in rows:
            word_dict.append(row)
        word_dict = dict(word_dict)
        retrieved_tokenizer['word_counts'] = word_dict

        return retrieved_tokenizer

    def captions_from_db(self):
        query = "SELECT image_name, captions_list from Captions"
        cursor.execute(query)
        rows = cursor.fetchall()
        cap_dict = []
        for row in rows:
            row = (row[0], row[1].split(', '))
            cap_dict.append(row)
        retrieved_captions = dict(cap_dict)

        return retrieved_captions

    def features_from_db(self):
        query = "SELECT image_name, feature from Features"
        cursor.execute(query)
        rows = cursor.fetchall()
        feat_dict = []
        for row in rows:
            np_array = row[1]
            np_array = np.frombuffer(np_array, dtype=np.float32)
            np_array = np.expand_dims(np_array, axis=0)
            row = (row[0], np_array)
            feat_dict.append(row)
        retrieved_features = dict(feat_dict)

        return retrieved_features
