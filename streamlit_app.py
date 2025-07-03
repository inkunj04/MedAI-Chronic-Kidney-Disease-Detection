import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import base64
import os
import smtplib
from email.message import EmailMessage

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="MedAI - CKD Analyzer", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.image("medai_logo.png", width=120)
    st.title("🧠 MedAI")
    st.markdown("""
**MedAI** is an AI-powered assistant for analyzing **Chronic Kidney Disease (CKD)**.

**Features**:
- ML-based prediction (RF, XGB, LR)
- LIME Explainability
- Auto Clinical Recommendations
- Doctor Notes 
- PDF Report & Email Integration
- Patient Similarity
- FAQ Chatbot + Feedback

Built by: Kunj Mehta
    """)

# ---------- Upload Dataset ----------
uploaded = st.file_uploader("📂 Upload CKD dataset (CSV)", type="csv")
if not uploaded:
    st.info("Please upload your dataset to proceed.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(" Dataset loaded!")

# ---------- Cleaning ----------
if 'id' in df: df.drop('id', axis=1, inplace=True)
df['classification'] = df['classification'].astype(str).str.strip().replace({'ckd': 1, 'notckd': 0})
binary_map = {'yes': 1, 'no': 0, 'present': 1, 'notpresent': 0, 'abnormal': 1, 'normal': 0, 'good': 1, 'poor': 0}
for col in ['rbc', 'pc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
    if col in df: df[col] = df[col].map(binary_map)
for col in ['pcv', 'wc', 'rc']:
    if col in df: df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

Xall = df.drop('classification', axis=1)
y = df['classification']
X = Xall.select_dtypes(include='number')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Field Full Forms ----------
full_forms = {"age":"Age (years)", "bp":"Blood Pressure (mm/Hg)", "sg":"Specific Gravity (urine)", "al":"Albumin (urine level)", "su":"Sugar (urine glucose)", "rbc":"Red Blood Cells (normal/abnormal)", "pc":"Pus Cell (normal/abnormal)", "pcc":"Pus Cell Clumps (present/notpresent)", "ba":"Bacteria (present/notpresent)", "bgr":"Blood Glucose Random (mg/dL)", "bu":"Blood Urea (mg/dL)", "sc":"Serum Creatinine (mg/dL)", "sod":"Sodium (mEq/L)", "pot":"Potassium (mEq/L)", "hemo":"Hemoglobin (g/dL)", "pcv":"Packed Cell Volume (%)", "wc":"White Blood Cell Count (cells/cumm)", "rc":"Red Blood Cell Count (millions/cumm)", "htn":"Hypertension (yes/no)", "dm":"Diabetes Mellitus (yes/no)", "cad":"Coronary Artery Disease (yes/no)", "appet":"Appetite (good/poor)", "pe":"Pedal Edema (yes/no)", "ane":"Anemia (yes/no)", "classification":"CKD Diagnosis"}


# ---------- Clinical Recommendations ----------
def generate_recommendations(data):
    recs = []
    if data.get("sc", 0) > 1.5:
        recs.append(" High creatinine: possible kidney dysfunction.")
    if data.get("hemo", 99) < 11:
        recs.append(" Low hemoglobin: anemia risk.")
    if data.get("bp", 0) > 140:
        recs.append(" High BP: monitor hypertension.")
    if data.get("bu", 0) > 50:
        recs.append(" Elevated urea: possible renal insufficiency.")
    if data.get("pot", 0) > 5.5:
        recs.append(" High potassium: cardiac risk, dietary change advised.")
    if data.get("sod", 0) < 135:
        recs.append("Low sodium: possible fluid imbalance.")
    return recs if recs else [" All clinical parameters within normal range."]

# ---------- PDF Generator ----------
def generate_pdf(data, diagnosis, confidence, notes, recs, sig_img=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "MedAI - CKD Diagnosis Report", ln=True, align='C')
    pdf.cell(200, 10, f"Prediction: {diagnosis}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, "Patient Data:", ln=True)
    for key, val in data.items():
        label = full_forms.get(key, key)
        pdf.cell(200, 10, f"{label} ({key}): {val}", ln=True)

    if notes:
        pdf.ln(5)
        pdf.multi_cell(200, 10, f"Doctor Notes:\n{notes}")

    if recs:
        pdf.ln(5)
        pdf.multi_cell(200, 10, "Clinical Recommendations:\n" + "\n".join(recs))


    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 EDA", "🧬 AutoML", "🎯 Predict", "📈 Sensitivity",
    "🤖 Chatbot", "📝 Feedback", "📧 Email Report"
])

# ---------- Tab 1 ----------
with tab1:
    st.header("📊 EDA")
    st.dataframe(df.head())
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# ---------- Tab 2 ----------
with tab2:
    st.header("🧬 AutoML Comparison")
    if st.button("Run AutoML"):
        clf = LazyClassifier()
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
        st.dataframe(models.sort_values("Accuracy", ascending=False))

# ---------- Tab 3 ----------

with tab3:
    st.header("🎯 CKD Prediction Wizard")
    consent = st.checkbox("✔️ Patient consent confirmed", value=True)
    if not consent:
        st.warning("Consent required.")
        st.stop()

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"{col}", value=float(X[col].median()), help=full_forms.get(col, ""))

    input_df = pd.DataFrame([user_input])
    model_name = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "XGBoost"])
    model = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    }[model_name]
    model.fit(X_train, y_train)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1] * 100
    st.success(f"Prediction: {'CKD' if prediction else 'No CKD'} ({proba:.2f}% confidence)")

    doctor_notes = st.text_area("🩺 Doctor Notes")
    recommendations = generate_recommendations(user_input)

    st.subheader("💊 Clinical Recommendations")
    for r in recommendations:
        st.markdown(f"- {r}")

    st.subheader("👥 Similar Patients")
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    _, idx = nn.kneighbors(input_df)
    st.dataframe(X.iloc[idx[0]])

    st.subheader("🧠 Explainability (LIME)")
    explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns.tolist(), class_names=["No CKD", "CKD"], mode="classification")
    exp = explainer.explain_instance(input_df.values[0], model.predict_proba, num_features=5)
    st.pyplot(exp.as_pyplot_figure())

    if st.button("📄 Generate PDF Report"):
        pdf_bytes = generate_pdf(user_input, "CKD" if prediction else "No CKD", proba, doctor_notes, recommendations)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="MedAI_Report.pdf">📥 Download PDF</a>', unsafe_allow_html=True)

# ---------- Tab 4 ----------
with tab4:
    st.header("📈 Sensitivity Analysis")
    sel = st.selectbox("Choose Feature", X.columns)
    val = st.slider(f"Change {sel}", float(X[sel].min()), float(X[sel].max()), float(X[sel].median()))
    temp = input_df.copy()
    temp[sel] = val
    new_proba = model.predict_proba(temp)[0][1] * 100
    st.write(f"Updated CKD Probability: **{new_proba:.2f}%**")

# ---------- Tab 5 ----------
with tab5:
    st.header("🤖 MedBot – CKD FAQ Chatbot")
    def medbot_response(q):
        q = q.lower()
        faqs = {
    # What is CKD / Basic Info
    "what is ckd": "CKD stands for Chronic Kidney Disease — a long-term condition where the kidneys gradually lose function over time.",
    "ckd meaning": "CKD means Chronic Kidney Disease — progressive damage to the kidneys.",
    "chronic kidney disease": "Chronic Kidney Disease is a long-term condition where your kidneys do not work effectively.",

    # Symptoms
    "ckd symptoms": "Common symptoms include fatigue, swelling (edema), nausea, poor appetite, high blood pressure, and frequent urination.",
    "how to know if i have kidney disease": "You may notice fatigue, swelling in feet/ankles, nausea, or foamy urine. A blood/urine test can confirm.",
    "signs of kidney failure": "Signs include swelling, decreased urine output, confusion, shortness of breath, and chest pain.",

    # Causes
    "causes of ckd": "Major causes are diabetes, high blood pressure, recurring kidney infections, and prolonged use of painkillers (NSAIDs).",
    "how do kidneys get damaged": "High BP, uncontrolled diabetes, infections, and overuse of medications like NSAIDs can damage kidneys.",

    # Prevention
    "how to prevent ckd": "Control blood pressure and diabetes, stay hydrated, avoid unnecessary painkillers, and eat a kidney-friendly diet.",
    "prevent kidney damage": "Avoid high-salt diets, smoking, alcohol, and stay active. Regular health checkups help too.",

    # Diagnosis
    "how is ckd diagnosed": "CKD is diagnosed using blood tests (eGFR, creatinine), urine tests (protein), and imaging studies (ultrasound).",
    "ckd test": "Blood tests like creatinine and eGFR, along with urine tests for protein, help diagnose CKD.",

    # Stages
    "ckd stages": "There are 5 stages of CKD. Stage 1 is mild, Stage 5 is kidney failure requiring dialysis or transplant.",
    "what are the stages of kidney disease": "Stage 1 to 5: based on eGFR. Stage 5 (eGFR <15) means kidney failure.",

    # Treatment
    "ckd treatment": "Treatment includes managing underlying conditions (like diabetes), medications, lifestyle changes, and in late stages, dialysis or transplant.",
    "how to treat kidney disease": "No cure, but progression can be slowed with diet, medications, and controlling blood pressure/sugar.",

    # Diet
    "ckd diet": "CKD patients are advised low-sodium, low-protein, and potassium/phosphorus-controlled diets. Consult a renal dietitian.",
    "foods to avoid in ckd": "Avoid salty foods, red meat, bananas, dairy, cola, and processed foods. Always consult your doctor.",

    # Dialysis
    "what is dialysis": "Dialysis is a treatment that removes waste, salt, and extra water to help your kidneys when they can’t do it anymore.",
    "do i need dialysis": "Dialysis is usually needed in Stage 5 CKD. Your doctor will decide based on tests like eGFR and symptoms.",

    # Lifestyle
    "can i exercise with ckd": "Yes, light to moderate exercise is usually safe and helpful unless advised otherwise by your doctor.",
    "is alcohol safe in ckd": "Alcohol should be limited or avoided. Always check with your nephrologist.",

    # Other
    "is ckd reversible": "CKD is usually not reversible, but with proper care, its progression can be slowed or stabilized.",
    "can i live long with ckd": "Yes, many people with CKD live long lives, especially with early diagnosis and proper management.",
    "difference between ckd and akd": "CKD is chronic and progresses over months/years. AKD (acute) occurs suddenly and may be reversible."
}

        for key, ans in faqs.items():
            if key in q:
                return ans
        return "🤖 I'm learning! Try rephrasing."
    user_q = st.text_input("Ask about CKD:")
    if st.button("Ask"):
        st.info(medbot_response(user_q))

# ---------- Tab 6 ----------
with tab6:
    st.header("📝 Feedback")
    actual = st.radio("Real Diagnosis", ["CKD", "No CKD"])
    correct = st.radio("Prediction Correct?", ["Yes", "No"])
    notes = st.text_area("Comments?")
    if st.button("Submit"):
        fb = pd.DataFrame([{
            "actual": actual,
            "prediction_correct": correct,
            "notes": notes
        }])
        fb.to_csv("feedback_log.csv", mode="a", header=not os.path.exists("feedback_log.csv"), index=False)
        st.success(" Feedback Submitted")

# ---------- Tab 7 ----------

with tab7:
    st.header("📧 Send Report via Email")

    def send_email_report(sender_email, app_password, receiver_email, subject, body, pdf_data):
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg.set_content(body)
            msg.add_attachment(pdf_data, maintype='application', subtype='pdf', filename='MedAI_Report.pdf')
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(sender_email, app_password)
                smtp.send_message(msg)
            return True, " Email sent!"
        except Exception as e:
            return False, f" Failed: {e}"

    with st.form("email_form"):
        st.subheader("📤 Send Report")
        sender_email = st.text_input("Sender Gmail")
        app_password = st.text_input("App Password", type="password")
        receiver_email = st.text_input("Recipient Email")
        subject = st.text_input("Email Subject", value="MedAI - CKD Diagnosis Report")
        message = st.text_area("Email Body", value="Dear Patient,\n\nPlease find your CKD report attached.\n\nRegards,\nMedAI Team")
        submitted = st.form_submit_button("Send Email")

        if submitted:
            if 'user_input' not in locals():
                st.warning("Please run prediction first.")
            else:
                pdf_data = generate_pdf(user_input, "CKD" if prediction else "No CKD", proba, doctor_notes, recommendations)
                success, feedback = send_email_report(sender_email, app_password, receiver_email, subject, message, pdf_data)
                st.info(feedback)
