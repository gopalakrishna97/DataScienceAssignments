from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    dependents = int(request.form['dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    applicant_income = int(request.form['applicant_income'])
    coapplicant_income = int(request.form['coapplicant_income'])
    loan_amount = int(request.form['loan_amount'])
    loan_amount_term = int(request.form['loan_amount_term'])
    credit_history = int(request.form['credit_history'])
    property_area = int(request.form['property_area'])

    # Make the loan eligibility prediction

    print([[gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area]])


    pred = model.predict([[gender,
                                 married,
                                 dependents,
                                 education,
                                 self_employed,
                                 applicant_income,
                                 coapplicant_income,
                                 loan_amount,
                                 loan_amount_term,
                                 credit_history,
                                 property_area]])

    if(pred[0]==1):
        prediction = "Eligible for the Loan"
    else:
        prediction = "Not Eligible for the Loan"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
