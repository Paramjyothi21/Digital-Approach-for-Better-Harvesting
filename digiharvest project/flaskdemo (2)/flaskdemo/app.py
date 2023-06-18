
from flask import Flask, render_template, request, Markup,redirect, url_for, session
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
import pickle
import sys
import datetime
from bs4 import BeautifulSoup
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath ='C:/Users/param jyothi/Desktop/digiharvest project/flaskdemo (2)/flaskdemo/Trained_model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")
def pred_pest(pest):
    
        test_image = load_img(pest, target_size = (64, 64)) # load image 
        print("@@ Got Image for prediction")
  
        test_image = img_to_array(test_image)# convert image to np array and normalize
        test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
        result = model.predict(test_image) # predict diseased palnt or not
        print('@@ Raw result = ', result)
  
        pred = np.argmax(result, axis=1)
        return pred
    

crop_recommendation_model = pickle.load(open('C:/Users/param jyothi/Desktop/digiharvest project/flaskdemo (2)/flaskdemo/RFmodel.pkl', "rb"))

yield_prediction_model = pickle.load(open('C:/Users/param jyothi/Desktop/digiharvest project/flaskdemo (2)/flaskdemo/RFYield.pkl', "rb"))

fpath = 'C:/Users/param jyothi/Desktop/digiharvest project/Plant-Seedlings-Classification-master/Plant-Seedlings-Classification-master/model_weight_Adam.hdf5'

model2 = load_model(fpath)
print(model2)

print("Model2 Loaded Successfully")

def pred_weed(weed):
    
        test_image = load_img(weed, target_size = (51, 51)) # load image 
        print("@@ Got Image for prediction")
  
        test_image = img_to_array(test_image)# convert image to np array and normalize
        test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
        result = model2.predict(test_image) 
        print('@@ Raw result = ', result)
  
        find = np.argmax(result, axis=1)
        return find

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '11113333'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Param2123@'
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)



@app.route("/")
# http://localhost:5000/pythonlogin/ - the following will be our login page, which will use both GET and POST requests
@app.route('/applogin/', methods=['GET', 'POST'])
def login():
   
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM account WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)
# http://localhost:5000/python/logout - this will be the logout page
@app.route('/applogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/applogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM account WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO account VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)
 # http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/applogin/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login')) 
# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/applogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM account WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))  

@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")


@app.route("/Costofcultivation.html")
def cultivation():
    return render_template("Costofcultivation.html")


@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/herbicides.html")
def herbicide():
    return render_template("herbicides.html")   

@app.route("/weed.html")
def weed():
    return render_template("weed.html")


@app.route("/Yieldprediction.html")
def yieldpred():
    return render_template("Yieldprediction.html")


@app.route("/Yieldresult")
def yieldresult():
    return render_template("Yieldresult.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('C:/Users/param jyothi/Desktop/digiharvest project/flaskdemo (2)/flaskdemo/static/user uploaded//', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
    
        if pred[0] == 0:
            pest_identified = 'aphids'
        elif pred[0] == 1:
            pest_identified = 'armyworm'
        elif pred[0] == 2:
            pest_identified = 'beetle'
        elif pred[0] == 3:
            pest_identified = 'bollworm'
        elif pred[0] == 4:
            pest_identified = 'earthworm'
        elif pred[0] == 5:
            pest_identified = 'grasshopper'
        elif pred[0] == 6:
            pest_identified = 'mites'
        elif pred[0] == 7:
            pest_identified = 'mosquito'
        elif pred[0] == 8:
            pest_identified = 'sawfly'
        elif pred[0] == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

@ app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')


@ app.route('/yield_prediction', methods=['POST'])
def  yield_prediction():
    if request.method == 'POST':
        crop = int(request.form['Crop'])
        state = int(request.form['State'])
        coca = int(request.form['Cost of Cultivation (`/Hectare) A2+FL'])
        cocb = float(request.form['Cost of Cultivation (`/Hectare) C2'])
        cop = float(request.form['Cost of Production (`/Quintal) C2'])
        ypq = float(request.form['Yield (Quintal/ Hectare)'])
        phcp = float(request.form['Per Hectare Cost Price'])
        data = np.array([[crop, state, coca, cocb, cop, ypq, phcp]])
        my_pred = yield_prediction_model.predict(data)
        newarr = np.array_split(my_pred, 1)
        return render_template('Yieldresult.html', costofcultivation=newarr[0][0][0],totalyield=newarr[0][0][1])
       

@app.route("/predictweed", methods=['GET', 'POST'])

def predictweed():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('C:/Users/param jyothi/Desktop/digiharvest project/flaskdemo (2)/flaskdemo/UPLOADING PHOTOS', filename)
        file.save(file_path)

        find = pred_weed(weed=file_path)
        
        if  find[0] == 0:
            weed_identified = 'Black-grass'
        elif find[0] == 1:
            weed_identified = 'Charlock'
        elif find[0] == 2:
            weed_identified = 'Cleavers'
        elif find[0] == 3:
            weed_identified = 'Common Chickweed'
        elif find[0] == 4:
            weed_identified = 'Common wheat'
        elif find[0] == 5:
            weed_identified = 'Fat Hen'
        elif find[0] == 6:
            weed_identified = 'Loose Silky-bent'
        elif find[0] == 7:
            weed_identified = 'Maize'
        elif find[0] == 8:
            weed_identified = 'Scentless Mayweed'
        elif find[0] == 9:
            weed_identified = 'Shepherds Purse'
        elif find[0] == 10:
            weed_identified = 'Small-flowered Cranesbill'
        elif find[0] == 11:
            weed_identified = 'Sugar beet'

        return render_template("result.html",weed_identified=weed_identified)
   

if __name__ == '__main__':
    app.run(debug=True)