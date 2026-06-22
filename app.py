import streamlit as st
import pandas as pd
import sqlite3
import joblib
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease ML App", layout="centered")

# ---------------- BASE64 BACKGROUND ----------------
# ---------------- BASE64 BACKGROUND ----------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 👉 Put any image in same folder (bg.jpg)
try:
    img = get_base64("bg.jpg")

    st.markdown(f"""
    <style>

    /* MAIN BACKGROUND FIX */
    .stApp {{
        background: url("data:image/jpg;base64,{img}") no-repeat center center fixed;
        background-size: 92% auto; /* Slightly zoomed out from original cover size */
    }}

    /* REMOVE STREAMLIT DARK/WHITE LAYERS */
    .main {{
        background: transparent !important;
    }}

    .block-container {{
        background: transparent !important;
    }}

    header {{
        background: transparent !important;
    }}

    footer {{
        background: transparent !important;
    }}

    /* OPTIONAL OVERLAY (for readability) */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.25);
        backdrop-filter: blur(6px); /* Blurs the background image */
        -webkit-backdrop-filter: blur(6px);
        z-index: -1;
    }}

    </style>
    """, unsafe_allow_html=True)

except:
    pass

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
h1, h2, h3 {
    color: #00FFFF;
    text-align: center;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stTextInput>div>div>input {
    border-radius: 10px;
}

.stNumberInput input {
    border-radius: 10px;
}

.css-1d391kg {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT,
    password TEXT,
    mobile TEXT,
    age INTEGER,
    source TEXT
)
""")
conn.commit()

# ---------------- SESSION ----------------
defaults = {
    "logged_in": False,
    "page": "login",
    "df": None,
    "model": None,
    "columns": None
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------- AUTH ----------------
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        if cursor.fetchone():
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "upload"
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

def register():
    st.title("📝 Register")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    mobile = st.text_input("Mobile Number")
    age = st.number_input("Age", 10, 100)

    source = st.selectbox(
        "How did you hear about us?",
        ["Instagram", "YouTube", "Friends", "Google", "Other"]
    )

    if st.button("Register"):
        if username and password and mobile:
            cursor.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                (username, password, mobile, age, source)
            )
            conn.commit()
            st.success("Registered Successfully ✅")
        else:
            st.error("Fill all fields ❌")



# ---------------- STEP 1 ----------------
def upload_page():
    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.write(df.head())

    if st.button("Next ➡️"):
        if st.session_state.df is not None:
            st.session_state.page = "clean"
            st.rerun()

# ---------------- STEP 2 ----------------
def clean_page():
    st.title(" Data Cleaning")

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")
        return

    if st.button("Clean Data"):
        df = df.drop_duplicates().dropna()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = LabelEncoder().fit_transform(df[col])

        st.session_state.df = df
        st.success("Data Cleaned ✅")

    st.write(df.head())

    
    if st.button("Next ➡️"):
        st.session_state.page = "train"
        st.rerun()
    if st.button("⬅️ Back"):
        st.session_state.page = "upload"
        st.rerun()


# ---------------- STEP 3 ----------------
def train_page():
    st.title("🤖 Train Model")

    df = st.session_state.df

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        X = df.drop(target, axis=1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        joblib.dump(model, "model.pkl")
        joblib.dump(X.columns.tolist(), "columns.pkl")

        st.session_state.model = model
        st.session_state.columns = X.columns.tolist()

        st.success(f"Accuracy: {acc:.2f}")

    
    if st.button("Next ➡️"):
        st.session_state.page = "load"
        st.rerun()
        
    if st.button("⬅️ Back"):
        st.session_state.page = "clean"
        st.rerun()


# ---------------- STEP 4 ----------------
def load_page():
    st.title("📦 Load Model")

    if st.button("Load Model"):
        try:
            st.session_state.model = joblib.load("model.pkl")
            st.session_state.columns = joblib.load("columns.pkl")
            st.success("Model Loaded ✅")
        except:
            st.error("Model not found")

    if st.button("Next ➡️"):
        st.session_state.page = "predict"
        st.rerun()
        
    if st.button("⬅️ Back"):
        st.session_state.page = "train"
        st.rerun()


# ---------------- STEP 5 ----------------
def predict_page():
    st.title("🔮 Prediction")

    model = st.session_state.model
    cols = st.session_state.columns

    inputs = []
    for col in cols:
        inputs.append(st.number_input(col, value=0.0))

    if st.button("Predict"):
        data = pd.DataFrame([inputs], columns=cols)
        result = int(model.predict(data)[0])

        if result == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk")
            
    if st.button("⬅️ Back"):
        st.session_state.page = "load"
        st.rerun()        

# ---------------- INFO PAGE ----------------
def info_page():
    st.title("❤️ Heart Disease Awareness")

    # 🔙 Back Button
    

    # ---------------- CAUSES ----------------
    st.header("⚠️ Causes (Detailed)")
    st.markdown("""
- Smoking & tobacco use damages arteries  
- High blood pressure increases heart strain  
- High cholesterol blocks blood flow  
- Diabetes damages blood vessels  
- Obesity leads to heart overload  
- Lack of exercise weakens heart muscles  
- Chronic stress increases risk  
- Poor sleep affects heart rhythm  
- Excess alcohol consumption  
- Genetic/family history  
- Air pollution exposure  
- Unhealthy processed food diet  
- Long screen time & no physical activity  
- Irregular eating habits  
""")

    # ---------------- INCREASE REASONS ----------------
    st.header("📈 Why cases increasing yearly?")
    st.markdown("""
- Sedentary lifestyle (more screen time)  
- Junk food consumption rising  
- Stress levels increasing in students & adults  
- Less physical activity  
- Urban pollution  
- Early-age hypertension & diabetes  
- Increased dependency on technology  
- Poor daily routines and sleep cycles  
""")

    # ---------------- PREVENTION ----------------
    st.header("🛡️ Prevention")
    st.markdown("""
✔ Eat fruits, vegetables & low-fat diet  
✔ Exercise 30 mins daily  
✔ Maintain healthy weight  
✔ Quit smoking completely  
✔ Limit salt & sugar intake  
✔ Sleep at least 7–8 hours  
✔ Regular BP & cholesterol check  
✔ Manage stress (yoga/meditation)  
✔ Drink enough water  
✔ Avoid processed/junk food  
✔ Regular medical checkups  
✔ Stay physically active (walk, sports)  
✔ Maintain a daily routine  
""")

    # ---------------- DAILY HABITS ----------------
    st.header("💡 Small Daily Habits That Make Big Difference")
    st.markdown("""
- Take stairs instead of lift  
- Walk after meals  
- Reduce mobile usage time  
- Drink more water instead of soft drinks  
- Spend time outdoors  
- Do light stretching daily  
""")

    # ---------------- MOTIVATION ----------------
   

    st.success("❤️ Take care of your heart today, so it takes care of you tomorrow.")

    st.info("💡 Small daily improvements lead to long-term health benefits.")

    st.warning("⚠️ Your future health depends on what you do today.")

    st.success("💪 A healthy heart is the key to a happy and active life.")

    st.markdown("---")
    st.markdown("### 🚀 Final Message")
    st.markdown("""
**Prevention is always better than cure.  
Start today. Stay consistent. Stay healthy. ❤️**
""")

    if st.button("⬅️ Back"):
        st.session_state.page = "upload"   # or last valid page fallback
        st.rerun()


# ---------------- MAIN ----------------
menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

if not st.session_state.logged_in:
    if menu == "Login":
        login()
    else:
        register()
else:
    st.sidebar.write(f"👤 {st.session_state.username}")

    # ✅ THIS BUTTON MUST BE ABOVE LOGOUT
    if st.sidebar.button("❤️ Causes & Prevention"):
        st.session_state.prev_page = st.session_state.page
        st.session_state.page = "info"
        st.rerun()

    # 🔴 LOGOUT BUTTON BELOW
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    pages = {
        "upload": upload_page,
        "clean": clean_page,
        "train": train_page,
        "load": load_page,
        "predict": predict_page,
        "info": info_page
    }

    pages[st.session_state.page]()
