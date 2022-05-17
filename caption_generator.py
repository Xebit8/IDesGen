import numpy as np
from pickle import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


tokenizer = load(open('train model\\IMPORTANT DATA\\tokenizer(NEW).p', 'rb'))
max_length = 34
model = load_model('train model\\models(NEW)\\model_20.h5')


class CaptionGenerator:
    def __init__(self):
        pass

    def extract_features(self, filename):
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        return feature

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate_desc(self, model, tokenizer, photo, max_length):
        input_text = 'startseq'
        for i in range(max_length):
            seq = tokenizer.texts_to_sequences([input_text])[0]
            seq = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([photo, seq], verbose=0)
            yhat = np.argmax(yhat)
            word = self.word_for_id(yhat, tokenizer)
            if word is None:
                break
            input_text += ' ' + word
            if word == 'endseq':
                break
        return input_text







