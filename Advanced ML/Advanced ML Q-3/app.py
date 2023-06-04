from flask import Flask, render_template, request
import pickle
import dask.dataframe as dd
import pandas as pd


# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input parameters
    tv = float(request.form['tv'])
    radio = float(request.form['radio'])
    newspaper = float(request.form['newspaper'])

    # Create input array for prediction
    input_data = dd.from_pandas(pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper']), npartitions=1)

    input_data = input_data.to_dask_array(lengths=True)
    # Make prediction
    prediction = model.predict(input_data)
    pred = prediction.compute()
    result = "{0:.2f}".format(pred[0])
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run()
