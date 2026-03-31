from unittest import result

from flask import Flask, render_template, request, flash, session, jsonify
from pymongo import MongoClient
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging


dataset = fetch_ucirepo(id=45) 
heart_disease = dataset.data.original
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.secret_key = 'kolhe'
client = MongoClient("mongodb+srv://pratikolhe1812_db_user:pratikolhe@cluster0.snhshb4.mongodb.net/")
db = client['patients']

# Make this a utility function, not an API route
def data_processing(age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal):
    app.logger.info("Processing input data...")

    # Convert inputs to float
    try:
        input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                      float(fbs), float(restecg), float(thalach), float(exang),
                      float(oldpeak), float(slope), float(ca), float(thal)]
    except ValueError:
        app.logger.error("Invalid input: All inputs must be numeric.")
        return {"error": "All inputs must be numeric.", "status": "error"}

    # Load dataset
    heart_data = pd.read_csv('./heart_disease_cleaned.csv')
    x = heart_data.drop(columns='num', axis=1)
    y = heart_data['num']

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Prediction
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    # Store in DB
    try:
        username = session.get('username')
        if not username:
            app.logger.error("Session missing or expired.")
            return {"error": "User not logged in", "status": "error"}
        tdate = pd.to_datetime('today').strftime('%Y-%m-%d')
        update_data = {
            "heart_data": input_data,
            "has_heart_disease": bool(prediction)
        }

        result = db.realusers.update_one(
            {"username": username},
            {"$set": update_data}
        )

        if result.matched_count == 0:
            app.logger.warning("No user found with username: %s", username)
            return {"error": "User not found in database", "status": "error"}

        if prediction == 0:
            app.logger.info("User %s is not at risk of heart disease.", username)
            flash('You are not at risk of heart disease. Keep maintaining a healthy lifestyle!', 'success')
            return {"prediction": "No Heart Disease", "status": "success"}
        else:
            app.logger.info("User %s is at risk of heart disease.", username)
            flash('You are at risk of heart disease. Please consult a doctor.', 'warning')
            return {"prediction": "Heart Disease Risk", "status": "warning"}

    except Exception as e:
        app.logger.error("Database update failed: %s", str(e))
        return {"error": "Database update failed", "status": "error"}

@app.route('/')
def home():
    print(heart_disease.isnull().sum())
    # heart_disease.to_csv('heart_disease.csv', index=False)
    return render_template('login.html')

@app.route('/appentry')
def appentry():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    user = db.realusers.find_one({'username': session.get('username')})
    if user:
        return render_template('dashboard.html', user=user)
    else:
        flash('You need to log in first!', 'danger')
        return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/api/predict_api', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        age      = request.form.get('age')
        gender   = request.form.get('gender')
        cp       = request.form.get('cp')
        trestbps = request.form.get('trestbps')
        chol     = request.form.get('chol')
        fbs      = request.form.get('fbs')
        restecg  = request.form.get('restecg')
        thalach  = request.form.get('thalach')
        exang    = request.form.get('exang')
        oldpeak  = request.form.get('oldpeak')
        slope    = request.form.get('slope')
        ca       = request.form.get('ca')
        thal     = request.form.get('thal')

        result = data_processing(age, gender, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak, slope, ca, thal)

        # ✅ Pass ALL inputs so the summary can render them
        return render_template('result.html', result=result,
                               age=age, gender=gender, cp=cp,
                               trestbps=trestbps, chol=chol, fbs=fbs,
                               restecg=restecg, thalach=thalach, exang=exang,
                               oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

@app.route('/api/api_register', methods=['GET', 'POST'])
def api_register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Both username and password are required!', 'danger')
            return render_template('register.html')
        elif username and password:
            existing_user = db.realusers.find_one({'username': username})
            if existing_user:
                flash('Username already exists!', 'danger')
                return render_template('register.html')
            else:
                db.realusers.insert_one({'username': username, 'password': password, })
                flash('Registration successful! You can now log in.', 'success')
                return render_template('login.html')

@app.route('/api/login',  methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Both username and password are required!', 'danger')
            return render_template('login.html')
        elif username and password:
            user = db.realusers.find_one({'username': username, 'password': password})
            if user:
                session['username'] = username
                flash('Login successful!', 'success')
                return render_template('dashboard.html', user=user)
            else:
                flash('Invalid username or password!', 'danger')
                return render_template('login.html')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
