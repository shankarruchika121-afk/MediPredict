from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "genz_health_secure_key"

# --- DATABASE CONFIGURATION (SQLite) ---
# FIXED: Changed lowercase 'healthcare.db' to match your actual file 'Healthcare.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Healthcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- USER MODEL (The Database Table) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False) # Hashed for security

# Create the database and tables automatically if they don't exist
with app.app_context():
    db.create_all()

# --- LOAD ML MODELS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models')
try:
    model_sym = pickle.load(open(os.path.join(model_path, "model_symptoms.pkl"), "rb"))
    le = pickle.load(open(os.path.join(model_path, "label_encoder.pkl"), "rb"))
    symptoms_list = pickle.load(open(os.path.join(model_path, "symptoms_list.pkl"), "rb"))
    model_diab = pickle.load(open(os.path.join(model_path, "model_diabetes.pkl"), "rb"))
    scaler_diab = pickle.load(open(os.path.join(model_path, "scaler_diabetes.pkl"), "rb"))
    print("✅ System Ready: Models Loaded & Database Initialized")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# --- AUTHENTICATION HELPER ---
def is_logged_in():
    return 'user_id' in session

# --- PAGE ROUTES ---

@app.route('/')
def home():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('predict.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form.get('username')
        pwd = request.form.get('password')
        
        if User.query.filter_by(username=uname).first():
            return "Error: Username already exists!", 400
        
        # Security: Optimized hashing for safer production environment package matching
        hashed_pwd = generate_password_hash(pwd)
        new_user = User(username=uname, password=hashed_pwd)
        
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form.get('username')
        pwd = request.form.get('password')
        
        user = User.query.filter_by(username=uname).first()
        
        if user and check_password_hash(user.password, pwd):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        
        return "Error: Invalid username or password", 401
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- MACHINE LEARNING API ENDPOINTS ---

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if not is_logged_in(): 
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.get_json()
    user_symptoms = data.get('symptoms', [])
    
    vec = np.zeros(len(symptoms_list))
    for s in user_symptoms:
        s_clean = s.strip().lower()
        if s_clean in symptoms_list:
            vec[symptoms_list.index(s_clean)] = 1
            
    probs = model_sym.predict_proba([vec])[0]
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    results = [
        {"disease": le.classes_[i], "confidence": round(float(probs[i]*100), 2)} 
        for i in top_3_indices
    ]
    return jsonify({"predictions": results})

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if not is_logged_in(): 
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.get_json().get('clinical_data')
    if not data:
        return jsonify({"error": "Missing data"}), 400
        
    scaled = scaler_diab.transform(np.array(data).reshape(1, -1))
    prediction = model_diab.predict(scaled)[0]
    
    return jsonify({"risk_score": "High Risk" if int(prediction) == 1 else "Low Risk"})

if __name__ == '__main__':
    app.run()