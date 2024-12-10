from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_model.sav', 'rb'))
parkinson_model = pickle.load(open('parkinson_model.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    diagnosis = ''
    if request.method == 'POST':
        # Retrieve input values from the form
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])

        # Predict
        prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diagnosis = 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

    return render_template('diabetes.html', diagnosis=diagnosis)


@app.route('/heart', methods=['GET', 'POST'])
def heart():
    diagnosis = ''
    if request.method == 'POST':
        # Retrieve input values from the form
        inputs = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal']),
        ]

        # Predict
        prediction = heart_model.predict([inputs])
        diagnosis = 'The person is having heart disease' if prediction[
                                                                0] == 1 else 'The person does not have any heart disease'

    return render_template('heart.html', diagnosis=diagnosis)


@app.route('/parkinson', methods=['GET', 'POST'])
def parkinson():
    diagnosis = ''
    if request.method == 'POST':
        # Retrieve input values from the form
        inputs = [
            float(request.form['fo']),
            float(request.form['fhi']),
            float(request.form['flo']),
            float(request.form['Jitter_percent']),
            float(request.form['Jitter_Abs']),
            float(request.form['RAP']),
            float(request.form['PPQ']),
            float(request.form['DDP']),
            float(request.form['Shimmer']),
            float(request.form['Shimmer_dB']),
            float(request.form['APQ3']),
            float(request.form['APQ5']),
            float(request.form['APQ']),
            float(request.form['DDA']),
            float(request.form['NHR']),
            float(request.form['HNR']),
            float(request.form['RPDE']),
            float(request.form['DFA']),
            float(request.form['spread1']),
            float(request.form['spread2']),
            float(request.form['D2']),
            float(request.form['PPE']),
        ]

        # Predict
        prediction = parkinson_model.predict([inputs])
        diagnosis = "The person has Parkinson's disease" if prediction[
                                                                0] == 1 else "The person does not have Parkinson's disease"

    return render_template('parkinson.html', diagnosis=diagnosis)


if __name__ == '__main__':
    app.run(debug=True)
