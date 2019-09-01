from flask import Flask, render_template, redirect, url_for, request
from pyrebase import pyrebase

app = Flask(__name__)

config = {
    "apiKey": "AIzaSyBCmoC7fFVKRsztfHL_HqmQo29ooLvRBpw",
    "authDomain": "hacknight-65e59.firebaseapp.com",
    "databaseURL": "https://hacknight-65e59.firebaseio.com",
    "projectId": "hacknight-65e59",
    "storageBucket": "hacknight-65e59.appspot.com",
    "messagingSenderId": "582741754484",
    "appId": "1:582741754484:web:801a11d057c78a8e"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect("/index")
    return render_template('login.html', error=error)

@app.route('/index')
def check():
    res = db.child("hr").get().val().values()
    return render_template('index.html', event=res)

@app.route('/employee', methods=['GET', 'POST'])
def employee():
    di = {}
    if request.method == 'POST':
        id = request.form['empid']
        di['empid'] = id
        di['Age'] = int(request.form['age'])
        di['Gender'] = request.form['gender']
        di['self_employed'] = request.form['self_employed']
        di['family_history'] = request.form['family_history']
        di['work_interfere'] = request.form['work_interfere']
        di['remote_work'] = request.form['remote_work']
        di['tech_company'] = request.form['tech_company']
        di['benefits'] = request.form['benefits']
        di['care_options'] = request.form['care_options']
        di['wellness_program'] = request.form['wellness_program']
        di['seek_help'] = request.form['seek_help']
        di['anonymity'] = request.form['anonymity']
        di['leave'] = request.form['leave']
        di['mental_health_consequence'] = request.form['mental_health_consequence']
        di['phys_health_consequence'] = request.form['phys_health_consequence']
        di['coworkers'] = request.form['coworkers']
        di['supervisor'] = request.form['supervisor']
        di['mental_health_interview'] = request.form['mental_health_interview']
        di['phys_health_interview'] = request.form['phys_health_interview']
        di['mental_vs_physical'] = request.form['mental_vs_physical']
        di['obs_consequence'] = request.form['obs_consequence']
        db.child("employee").child(id).push(di)
        return render_template('login.html')
    return render_template('indexEmployee.html')