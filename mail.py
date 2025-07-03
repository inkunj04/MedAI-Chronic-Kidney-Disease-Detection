import streamlit as st
import smtplib
from email.message import EmailMessage
import pandas as pd
import numpy as np
from io import BytesIO
from fpdf import FPDF
import base64

st.set_page_config(page_title="MedAI - Email Report Sender", layout="wide")

st.title("📧 Email CKD Prediction Report - MedAI")

# Sample dummy data (replace with actual diagnosis input in real usage)
user_data = {
    "age": 55,
    "bp": 80,
    "sg": 1.02,
    "al": 1,
    "su": 0,
    "bgr": 130,
    "bu": 40,
    "sc": 1.5,
    "sod": 140,
    "pot": 4.5,
    "hemo": 12,
    "pcv": 40,
    "wbcc": 7000,
    "rbcc": 4.5,
}
prediction = "CKD"
confidence = 94.56

# ----------------------------------
# Generate PDF Report
def generate_pdf(data, diagnosis, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "MedAI - CKD Diagnosis Report", ln=True, align='C')
    pdf.cell(200, 10, f"Prediction: {diagnosis}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)

    for key, value in data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# ----------------------------------
# Email Sending Function
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

        return True, "✅ Email sent successfully!"
    except Exception as e:
        return False, f"❌ Failed to send email: {e}"

# ----------------------------------
# Email Form
with st.form("email_form"):
    st.subheader("📤 Send Report via Email")
    sender_email = st.text_input("Your Gmail Address")
    app_password = st.text_input("App Password (16-digit)", type="password")
    receiver_email = st.text_input("Recipient's Email")
    subject = st.text_input("Email Subject", value="MedAI - CKD Diagnosis Report")
    message = st.text_area("Email Body", value="Dear Patient,\n\nPlease find your CKD diagnostic report attached.\n\nRegards,\nMedAI Team")

    submitted = st.form_submit_button("📨 Send Email")
    if submitted:
        pdf_content = generate_pdf(user_data, prediction, confidence)
        success, feedback = send_email_report(sender_email, app_password, receiver_email, subject, message, pdf_content)
        st.info(feedback)
