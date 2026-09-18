# LoanLens — "Look Beyond the Loan"
### AI-Powered Legitimate Lender Verification & Scam Detection System

LoanLens is an end-to-end consumer safety and regulatory compliance platform designed to verify digital lenders, detect predatory loan apps, analyze suspicious loan messages/T&Cs, and evaluate excessive app permissions.

---

## 🎯 Core Features Implemented

1. **RBI Lender Verification (Stage 1)**
   - Searches lender against the official RBI-registered NBFC database (~8,500+ records).
   - Validates official registered company name, Corporate Identification Number (CIN), classification, RBI regulatory layer (Base/Middle/Upper), regional office, and address.
   - Shows 🟢 **RBI Registered** if match found, with smart alias & fuzzy name normalization.

2. **Flagged / Banned Lender Detection (Stage 1)**
   - Registry of known illegal, banned, extortionate, or impersonating loan apps.
   - Instant 🔴 **Flagged / High Risk** result with reason, date, and regulatory advisory source.

3. **Unknown Lender Detection**
   - Lenders not found in RBI or Flagged databases are marked 🟡 **Unverified**.
   - **Automatically opens Stage 2** for deep scrutiny of loan messages, T&Cs, and requested Android permissions.

4. **Loan Message / T&C / Privacy Policy Analysis (Stage 2)**
   - Scans text for 10 suspicious pattern categories:
     - Upfront processing & insurance fees
     - Guaranteed / instant approval claims
     - High-pressure urgency tactics
     - Illegal recovery & defamation threats
     - Fake RBI / Government scheme claims
     - Hidden charges & daily penalties
     - Unfair contract & unilateral modification clauses
     - Suspicious privacy & third-party data sharing terms
     - Suspicious / direct APK download links
     - Excessive data collection demands

5. **Permission Risk Analysis (RBI Compliance)**
   - Evaluates Android app permissions according to RBI Digital Lending Guidelines (2022):
     - **Prohibited / High Risk**: Contacts List, Call Logs, Photo Gallery / Files, Telephony / Phone State, Microphone (outside Video KYC), Device Administrator.
     - **Controlled / Medium Risk**: SMS (OTP only), Location (one-time KYC), Camera (live selfie), Storage.

6. **ML-Based Risk Classification & Model Comparison**
   - TF-IDF feature vector extraction with multi-model comparison:
     - **LinearSVC** (Primary Pipeline): 87.5% Accuracy, 95.2% Recall, 90.1% F1-Score.
     - **Logistic Regression**: 86.5% Accuracy, 96.8% Recall, 89.6% F1-Score.
     - **Multinomial Naive Bayes**: 88.5% Accuracy, 96.8% Recall, 90.9% F1-Score.
   - Classifies text into **Legitimate**, **Suspicious**, or **High Risk** with a confidence score.

7. **Explainable AI**
   - Does not just output "Scam". Extracts exact quoted snippets with rationale:
     - 🚩 *"Pay ₹1,499 processing & security fee before disbursement"* $\rightarrow$ Upfront fee violation.
     - 🚩 *"100% guaranteed approval within 5 minutes"* $\rightarrow$ Regulatory appraisal violation.
     - 🚩 *"Allow permission to access your contacts list and photo gallery"* $\rightarrow$ Prohibited permission demand.
     - 🚩 *"Contact your family members and employer"* $\rightarrow$ Criminal harassment threat.

8. **Composite Risk Score (0–100)**
   - Aggregates RBI verification status, flagged list hits, message red flags, ML confidence, and permission penalties.
   - Outputs:
     - 🟢 **Low Risk (0–34)**
     - 🟡 **Suspicious / Needs Verification (35–64)**
     - 🔴 **High Risk / Potential Scam (65–100)**

9. **Detailed Results Dashboard**
   - Animated SVG Risk Gauge, factor point breakdown, stage indicators, highlighted quotes, permission breakdown, and full summary report with 1-click clipboard export.

10. **Actionable User Safety Recommendations**
    - Contextual advice, RBI reporting guidance, and direct links to National Cybercrime Helpline (1930 / cybercrime.gov.in) and official RBI portals.

---

## 🚀 Demo Flow

$$\text{Enter Lender} \longrightarrow \text{RBI Check} \longrightarrow \text{Flagged List Check} \longrightarrow \text{If Unknown} \longrightarrow \text{T\&C / Message NLP} \longrightarrow \text{Permissions} \longrightarrow \text{ML Classifier} \longrightarrow \text{Explainable Risk Verdict}$$

---

## 🛠️ Architecture & Dual-Mode Deployment

LoanLens features a **zero-downtime hybrid architecture**:
- **Live Mode**: Calls Flask API (`app.py`) or Vercel Serverless Function (`api/index.py`).
- **Standalone Mode**: Built-in client-side engine (`frontend/js/api.js`) featuring embedded sample registry, regex NLP rules, and ML decision heuristics to guarantee 100% demo reliability anywhere.

```
loanlens/
├── app.py                     # Flask REST API backend
├── rbi_lookup.py              # RBI dataset search & fuzzy matching
├── train_model.py             # Model training & comparison script
├── pipeline.py                # CSV batch scoring pipeline
├── model.pkl / tfidf.pkl      # Pre-trained ML artifacts
├── metrics.json               # Model benchmark statistics
├── RBI_NBFC_final.csv         # RBI NBFC Master List (~8,500+ records)
├── data/
│   ├── flagged_lenders.json   # Known flagged & banned loan app registry
│   └── red_flag_rules.json    # Configurable suspicious pattern rules
├── frontend/
│   ├── index.html             # Landing page
│   ├── verify.html            # Core verification & dashboard page
│   ├── css/style.css          # Glassmorphism design system
│   └── js/api.js              # Universal hybrid API & client engine
├── api/
│   └── index.py               # Vercel serverless function entrypoint
├── vercel.json                # Vercel deployment configuration
└── requirements.txt           # Python dependencies
```

---

## 💻 Local Setup & Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aasthasingla15/loanlens.git
   cd loanlens
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**:
   ```bash
   python app.py
   ```
   Open `http://localhost:5001` in your browser.

---

## ☁️ Vercel Deployment

1. Install Vercel CLI: `npm install -g vercel` (or connect your GitHub repository on [vercel.com](https://vercel.com)).
2. Deploy directly:
   ```bash
   vercel
   ```
   `vercel.json` and `api/index.py` are already pre-configured.

---

## ⚖️ Disclaimer
LoanLens is an AI-assisted consumer awareness and risk indicator tool. It does not constitute legal or financial advice. Always verify lending institutions directly on [rbi.org.in](https://www.rbi.org.in).
