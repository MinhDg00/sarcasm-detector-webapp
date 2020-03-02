from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


app = Flask(__name__)

# IMAGE_FOLDER = os.path.join('static', 'img_pool')
# app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model
    # load the pre-trained Keras model
    model = load_model('model.h5')

# Sarcasm Detector 
@app.route('/')
def webpage():
    return render_template("webpage.html")

@app.route('/predict', methods = ['POST', "GET"])
def predict():
        # hyperparemeter
    vocab_size = 10000
    max_length = 32
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = 'Unknown'
    sarcasm = ''

        # get text & preprocess text
    if request.method == 'POST':
        text = request.form['Sentence']        
        tokenizer = Tokenizer(num_words = vocab_size,oov_token = oov_tok)
        tokenizer.fit_on_texts(text)
        text = tokenizer.texts_to_sequences(text)
        train = pad_sequences(text, maxlen = max_length, 
                               padding = padding_type, truncating = trunc_type)

         
    #vector = np.array([x_test.flatten()])
        sarcasm = model.predict_classes(train)[0][0]
        if sarcasm == 0:
            return render_template('webpage.html', pred = 'No Sarcasm Detected')
        else:
            return render_template('webpage.html', pred = 'Sarcasm Detected')
#########################Code for Sentiment Analysis

if __name__ == "__main__":
    init()
    app.run(debug = True)
