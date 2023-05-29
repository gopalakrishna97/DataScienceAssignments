from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('music_feature_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the labelencoder
with open('labelencoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    tempo = int(request.form['tempo'])
    beats = int(request.form['beats'])
    chroma_stft = int(request.form['chroma_stft'])
    rmse = int(request.form['rmse'])
    spectral_centroid = int(request.form['spectral_centroid'])
    spectral_bandwidth = int(request.form['spectral_bandwidth'])
    rolloff = int(request.form['rolloff'])
    zero_crossing_rate = int(request.form['zero_crossing_rate'])
    mfcc1 = int(request.form['mfcc1'])
    mfcc2 = int(request.form['mfcc2'])
    mfcc3 = int(request.form['mfcc3'])
    mfcc4 = int(request.form['mfcc4'])
    mfcc5 = int(request.form['mfcc5'])
    mfcc6 = int(request.form['mfcc6'])
    mfcc7 = int(request.form['mfcc7'])
    mfcc8 = int(request.form['mfcc8'])
    mfcc9 = int(request.form['mfcc9'])
    mfcc10 = int(request.form['mfcc10'])
    mfcc11 = int(request.form['mfcc11'])
    mfcc12 = int(request.form['mfcc12'])
    mfcc13 = int(request.form['mfcc13'])
    mfcc14 = int(request.form['mfcc14'])
    mfcc15 = int(request.form['mfcc15'])
    mfcc16 = int(request.form['mfcc16'])
    mfcc17 = int(request.form['mfcc17'])
    mfcc18 = int(request.form['mfcc18'])
    mfcc19 = int(request.form['mfcc19'])
    mfcc20 = int(request.form['mfcc20'])

    # Make the loan eligibility prediction
    print([[tempo,
            beats,
            chroma_stft,
            rmse,
            spectral_centroid,
            spectral_bandwidth,
            rolloff,
            zero_crossing_rate,
            mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,
            mfcc17,mfcc18,mfcc19,mfcc20
            ]])


    pred = model.predict([[tempo,
            beats,
            chroma_stft,
            rmse,
            spectral_centroid,
            spectral_bandwidth,
            rolloff,
            zero_crossing_rate,
            mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,
            mfcc17,mfcc18,mfcc19,mfcc20
            ]])


    print(pred)
    genre = encoder.inverse_transform(pred)


    return render_template('index.html', prediction=genre)

if __name__ == '__main__':
    app.run(debug=True)
