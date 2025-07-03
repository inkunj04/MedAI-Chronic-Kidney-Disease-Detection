# 🧠 MedAI – Chronic Kidney Disease Detection System

An intelligent AI-powered assistant that predicts the **presence of Chronic Kidney Disease (CKD)** using clinical parameters. It includes **ML-based predictions**, **LIME explainability**, **auto clinical suggestions**, **patient similarity**, **PDF report generation**, and **Gmail email integration** – all through a modern Streamlit UI.

> 🎓 Built with love by **Kunj Mehta**  
> 🔍 Domain: AI/ML + Healthcare | Interface: Streamlit | Status: ✅ Completed  

---

## 🧠 Problem Statement

Early detection of Chronic Kidney Disease (CKD) is crucial to prevent life-threatening complications. However, traditional diagnosis methods are often manual, inconsistent, and dependent on expert availability.

> **MedAI** offers a smart, automated, and explainable approach to **predict CKD diagnosis**, assist doctors with **clinical insights**, and generate **professional reports** in real-time.

---

## 🚀 Key Features

| Feature | Description |
|--------|-------------|
| 🔮 **Real-time CKD Prediction** | Predicts CKD from 24+ medical parameters using ML (RF, XGBoost, Logistic Regression). |
| 🧠 **LIME Explainability** | Provides transparent ML explanations for each patient case. |
| 💊 **Auto Clinical Recommendations** | Flags critical vitals and provides medical suggestions. |
| 🩺 **Doctor Notes** | Add custom notes directly to the report. |
| 📄 **PDF Report Generator** | Generates diagnosis report with summary, vitals, and notes. |
| 📧 **Gmail Email Integration** | Sends PDF report via Gmail SMTP using App Password. |
| 🧬 **Similar Patient Finder** | Finds 3 nearest patient records using Nearest Neighbors. |
| 🤖 **FAQ Chatbot** | Built-in CKD awareness bot with 25+ real-world FAQs. |
| 📝 **Feedback Module** | Log real vs predicted data to improve future model quality. |

---

## 🧰 Tech Stack

| Layer | Tools Used |
|------|------------|
| 🧠 ML Models | Random Forest, Logistic Regression, XGBoost |
| 🧪 Explainability | LIME (Local Interpretable Model-Agnostic Explanations) |
| 📊 Data Analysis | Pandas, Seaborn, Matplotlib |
| 🧾 Reporting | FPDF (PDF generation), smtplib (Email) |
| 🖥 Interface | Streamlit |
| 🔁 Misc | LazyPredict, Sklearn, NumPy |

---

## 📖 Project Explanation

### 🎯 Objective

The **MedAI** system aims to assist healthcare professionals in **detecting Chronic Kidney Disease (CKD)** using patient vitals. It combines machine learning predictions, transparency (LIME), and clinical logic to generate insightful, actionable reports for patient care.

---

### 🧠 How It Works

1. **Input Collection:**
   - Upload any CKD dataset (CSV).
   - Select model and enter 20+ clinical vitals (e.g., hemoglobin, blood urea, serum creatinine).

2. **Prediction & Recommendations:**
   - Predicts CKD presence and shows % confidence.
   - Generates clinical suggestions for risky parameters.

3. **Explanation & Analysis:**
   - Uses LIME to visually explain top 5 contributing features.
   - Identifies 3 most similar patients using Nearest Neighbors.

4. **Documentation:**
   - Doctor can add notes manually.
   - PDF summary of the diagnosis is generated.
   - Report can be emailed instantly via Gmail.

5. **Bonus Tools:**
   - Built-in chatbot answers common CKD-related questions.
   - Feedback tab logs misclassifications to a local CSV for review.

---

### ✅ Why It Matters

- Saves **diagnostic time** and improves consistency.
- Offers **explainable AI** insights for non-technical users (doctors).
- Enhances medical workflows with **automated reporting**.
- Makes healthcare AI **accessible, transparent, and deployable**.

---

## 📁 Project Structure

```bash
📦 MedAI - CKD Detection System
│
├── 📜 streamlit_app.py            # Main Streamlit application
├── 📁 models/                     # (optional) Pretrained model storage
├── 📓 symp.ipynb                  # Notebook for feature analysis (EDA, trials)
├── 📄 feedback_log.csv            # Feedback tracking file
├── 🖼️ medai_logo.png              # Project logo for sidebar
└── README.md                      # Project documentation
