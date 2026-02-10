from flask import Flask, request, jsonify, session
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from functools import wraps
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json

# Google OAuth imports
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
    print("Warning: Google auth libraries not installed. Run: pip install google-auth google-auth-oauthlib")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID_HERE')

# Configure CORS to allow requests from the frontend
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# Email configuration
EMAIL_SENDER = "churnpred4@gmail.com"
EMAIL_PASSWORD = "CHURN1234"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_email TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction_result TEXT NOT NULL,
            probability REAL NOT NULL,
            email_sent INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# Initialize database on startup
init_db()

# Load the trained model
try:
    model = joblib.load('rf_churn_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define feature columns
feature_cols = [
    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear',
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
    'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
    'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
    'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
    'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI', 'Gender_Male',
    'MaritalStatus_Married', 'MaritalStatus_Single', 'PreferedOrderCat_Grocery',
    'PreferedOrderCat_Laptop & Accessory', 'PreferedOrderCat_Mobile',
    'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others'
]

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print(f"Login attempt: {data}")
    
        username = data.get('username')
        password = data.get('password')
        
        # Accept any username and password
        if username and password:
            session.permanent = True
            session['logged_in'] = True
            session['username'] = username
            session['auth_method'] = 'traditional'
            return jsonify({'success': True, 'username': username})
        
        return jsonify({'success': False, 'message': 'Username and password are required'}), 401

    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Verify Google OAuth token and create session"""
    if not GOOGLE_AUTH_AVAILABLE:
        return jsonify({'success': False, 'message': 'Google authentication not available'}), 503
    
    try:
        data = request.get_json()
        token = data.get('credential')
        
        if not token:
            return jsonify({'success': False, 'message': 'No token provided'}), 400
        
        # Verify the token with Google
        try:
            idinfo = id_token.verify_oauth2_token(
                token, 
                google_requests.Request(), 
                GOOGLE_CLIENT_ID
            )
            
            # Token is valid, extract user info
            user_email = idinfo.get('email')
            user_name = idinfo.get('name')
            user_picture = idinfo.get('picture')
            
            # Create session
            session.permanent = True
            session['logged_in'] = True
            session['username'] = user_name
            session['email'] = user_email
            session['picture'] = user_picture
            session['auth_method'] = 'google'
            
            print(f"Google login successful: {user_email}")
            
            return jsonify({
                'success': True,
                'username': user_name,
                'email': user_email,
                'picture': user_picture
            })
            
        except ValueError as e:
            # Invalid token
            print(f"Invalid Google token: {str(e)}")
            return jsonify({'success': False, 'message': 'Invalid token'}), 401
            
    except Exception as e:
        print(f"Google auth error: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('email', None)
    session.pop('picture', None)
    session.pop('auth_method', None)
    return jsonify({'success': True})

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    if 'logged_in' in session:
        return jsonify({
            'authenticated': True,
            'username': session.get('username'),
            'email': session.get('email'),
            'picture': session.get('picture'),
            'auth_method': session.get('auth_method', 'traditional')
        })
    return jsonify({'authenticated': False})

def send_prediction_email(user_email, prediction_result, probability, input_data):
    """Send prediction results to user via email"""
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'ChurnAI Prediction Result: {prediction_result}'
        msg['From'] = EMAIL_SENDER
        msg['To'] = user_email
        
        # Create HTML email body
        html = f"""
        <html>
          <head></head>
          <body>
            <h2 style="color: #4A90E2;">ChurnAI Prediction Results</h2>
            <p>Hello,</p>
            <p>Your customer churn prediction has been completed. Here are the results:</p>
            
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0;">
              <h3 style="color: {'#E74C3C' if prediction_result == 'Churn' else '#27AE60'};">
                Prediction: {prediction_result}
              </h3>
              <p><strong>Probability:</strong> {probability:.2%}</p>
            </div>
            
            <h3>Input Parameters:</h3>
            <table style="border-collapse: collapse; width: 100%;">
              <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Parameter</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
              </tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">Tenure</td><td style="border: 1px solid #ddd; padding: 8px;">{input_data.get('Tenure', 'N/A')}</td></tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">City Tier</td><td style="border: 1px solid #ddd; padding: 8px;">{input_data.get('CityTier', 'N/A')}</td></tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">Warehouse To Home</td><td style="border: 1px solid #ddd; padding: 8px;">{input_data.get('WarehouseToHome', 'N/A')}</td></tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">Hour Spend On App</td><td style="border: 1px solid #ddd; padding: 8px;">{input_data.get('HourSpendOnApp', 'N/A')}</td></tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">Satisfaction Score</td><td style="border: 1px solid #ddd; padding: 8px;">{input_data.get('SatisfactionScore', 'N/A')}</td></tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">Complain</td><td style="border: 1px solid #ddd; padding: 8px;">{'Yes' if input_data.get('Complain') == 1 else 'No'}</td></tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">Order Count</td><td style="border: 1px solid #ddd; padding: 8px;">{input_data.get('OrderCount', 'N/A')}</td></tr>
              <tr><td style="border: 1px solid #ddd; padding: 8px;">Cashback Amount</td><td style="border: 1px solid #ddd; padding: 8px;">{input_data.get('CashbackAmount', 'N/A')}</td></tr>
            </table>
            
            <p style="margin-top: 20px;">Thank you for using ChurnAI!</p>
            <p style="color: #888; font-size: 12px;">This is an automated message. Please do not reply to this email.</p>
          </body>
        </html>
        """
        
        # Attach HTML content
        msg.attach(MIMEText(html, 'html'))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"Email sent successfully to {user_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        
        # Get user email - prioritize session email (from Google auth) over form email
        user_email = session.get('email') or data.get('email')
        if not user_email:
            return jsonify({'error': 'Email address is required'}), 400
        
        # Prepare input data
        input_data = {
            'Tenure': float(data['Tenure']),
            'CityTier': int(data['CityTier']),
            'WarehouseToHome': float(data['WarehouseToHome']),
            'HourSpendOnApp': float(data['HourSpendOnApp']),
            'NumberOfDeviceRegistered': int(data['NumberOfDeviceRegistered']),
            'SatisfactionScore': int(data['SatisfactionScore']),
            'NumberOfAddress': int(data['NumberOfAddress']),
            'Complain': int(data['Complain']),
            'OrderAmountHikeFromlastYear': float(data['OrderAmountHikeFromlastYear']),
            'CouponUsed': float(data['CouponUsed']),
            'OrderCount': float(data['OrderCount']),
            'DaySinceLastOrder': float(data['DaySinceLastOrder']),
            'CashbackAmount': float(data['CashbackAmount']),
            'PreferredLoginDevice': data['PreferredLoginDevice'],
            'PreferredPaymentMode': data['PreferredPaymentMode'],
            'Gender': data['Gender'],
            'MaritalStatus': data['MaritalStatus'],
            'PreferedOrderCat': data['PreferedOrderCat']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encoding
        input_df_encoded = pd.get_dummies(input_df, columns=[
            'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'MaritalStatus', 'PreferedOrderCat'
        ])

        # Ensure all required columns
        for col in feature_cols:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0

        # Reorder columns
        input_df_encoded = input_df_encoded[feature_cols]
        input_df_encoded = input_df_encoded.astype(int)

        # Make prediction
        prediction = model.predict(input_df_encoded)[0]
        probability = model.predict_proba(input_df_encoded)[0][1]

        result = 'Churn' if prediction == 1 else 'No Churn'
        
        # Save to database
        timestamp = datetime.now().isoformat()
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Send email
        email_sent = send_prediction_email(user_email, result, probability, input_data)
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, user_email, input_data, prediction_result, probability, email_sent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, user_email, json.dumps(input_data), result, float(probability), 1 if email_sent else 0))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'prediction': result,
            'probability': float(probability),
            'probability_formatted': f'{probability:.2%}',
            'email_sent': email_sent
        })


    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5051)