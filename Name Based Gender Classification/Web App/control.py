from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
import numpy as np

app = Flask(__name__)

M = pickle.load(open('../trainedModel/model.pkl', 'rb'))
N = pickle.load(open('../normalisation/normalized_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    gender = predict_gender(name)  # Implement your gender prediction logic here
    return render_template('result.html', name=name, gender=gender)


def transform(name):
    
    # name : string
    ## Applying one hot encoding
    name = name.strip()[0]
    x = np.array([name])
    x = np.char.lower(x)

    name = np.zeros((23, 26))
    for i in range(len(x[0])):
        if x[0][i] >= 'a' and x[0][i] <= 'z':
            index = ord(x[0][i]) - ord('a')
            name[i, index] = 1

    return name

def predict_gender(name):
    valid_input = transform(name)
    valid_input = np.array([valid_input])
    
    x = N(valid_input).numpy()
    
    x = x.reshape(1, 23*26)

    yhat = M.predict(x)
    yhat = tf.nn.sigmoid(yhat)

    return "Female" if yhat >= 0.5 else  "Male"
    
    return "Female" if yhat >= 0.5 else  "Male"

if __name__ == "__main__":
    app.run(debug=True)