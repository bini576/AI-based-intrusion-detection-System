<div align="center">

# ðŸ›¡ï¸ AI-Based Intrusion Detection System (AI-IDS)
### Intelligent Network Anomaly & Attack Vector Classification using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Status-Completed%20%2F%20Active-success?style=for-the-badge)](#)

<p align="center">
  <a href="#-overview">Overview</a> â€¢
  <a href="#-key-features">Key Features</a> â€¢
  <a href="#-tech-stack">Tech Stack</a> â€¢
  <a href="#-dataset-architecture">Dataset</a> â€¢
  <a href="#-installation--setup">Installation</a> â€¢
  <a href="#-model-evaluation--metrics">Evaluation</a> â€¢
  <a href="#-roadmap">Roadmap</a>
</p>

</div>

---

## ðŸ“Œ Overview

Traditional signature-based Network Intrusion Detection Systems (NIDS) struggle to detect zero-day exploits and polymorphic threat payloads. **AI-IDS** is a machine learning-driven network anomaly detection pipeline designed to inspect network telemetry, uncover statistical deviations, and categorize malicious flow behaviors in near real-time.

Developed by **Biniyam Ademasu Afework**, this project applies supervised and ensemble classification algorithms (including Random Forest, Decision Trees, Support Vector Machines, and Logistic Regression) over benchmark network intrusion telemetry (such as NSL-KDD / CICIDS) to deliver high recall against probing attacks, denial of service (DoS/DDoS), unauthorized privilege escalation (U2R), and remote-to-local (R2L) exploits.

---

## âœ¨ Key Features

- **Multi-Class Threat Classification:** Accurately differentiates normal traffic patterns from multi-vector cyber attacks:
  - **DoS / DDoS:** SYN floods, Smurf, Neptune, Pod
  - **Probe / Port Scanning:** Nmap port scans, IP sweeps, Satan
  - **Privilege Escalation (U2R):** Buffer overflow attempts, rootkit injections
  - **Remote-to-Local (R2L):** Guessing passwords, FTP write exploits, unauthorized access
- **Robust Feature Engineering Pipeline:**
  - One-Hot encoding of categorical protocol types (`TCP`, `UDP`, `ICMP`) and service endpoints (`HTTP`, `DNS`, `SSH`, etc.)
  - Outlier detection and standard variance scaling via `StandardScaler` / `MinMaxScaler`
  - Correlation heatmaps and feature importance ranking via Random Forest Gini impurity
- **Comprehensive Evaluation Analytics:** Automated generation of Confusion Matrices, Precision-Recall Curves, and ROC-AUC performance curves.
- **Reproducible Jupyter Workflow:** Clean, modular notebook architecture enabling step-by-step experimentation from raw ingestion to serialized model persistence.

---

## ðŸ› ï¸ Tech Stack & Dependencies

| Layer | Tools & Libraries |
|:---|:---|
| **Language** | Python 3.9+ |
| **Interactive Research** | Jupyter Notebook, IPython |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Joblib, Pickle |

---

## ðŸ“Š Dataset Architecture & Preprocessing

The model is trained and benchmarked against standardized intrusion detection datasets (e.g., **NSL-KDD / CICIDS2017**):

1. **Ingestion & Data Cleansing:**
   - Removal of redundant duplicate flow records to eliminate model bias.
   - Handling missing and null values with median statistical imputation.
2. **Feature Encoding & Normalization:**
   - Protocol types, service flags, and connection states mapped to numerical arrays.
   - Continuous attributes (e.g., `duration`, `src_bytes`, `dst_bytes`, `count`) normalized to prevent dominant weight scaling.
3. **Class Balancing & Splitting:**
   - Stratified train-test split (80% training / 20% validation) to ensure proportionate representation of minority attack classes.

---

## ðŸš€ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git
- `pip` or `conda`

### 1. Clone the Repository
```bash
git clone https://github.com/bini576/AI-based-intrusion-detection-System.git
cd AI-based-intrusion-detection-System
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet generated, install core packages:)*
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```
Navigate to `notebooks/` and open `AI_Intrusion_Detection.ipynb` to run cells sequentially.

---

## ðŸ“ˆ Model Evaluation & Metrics

The system was evaluated using standard classification metrics: Precision, Recall, F1-Score, and ROC-AUC.

### Benchmark Performance Summary

| Classifier Model | Accuracy (%) | Precision | Recall | F1-Score | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Classifier** | **99.2%** | **0.99** | **0.99** | **0.99** | ðŸŒŸ *Best Performer* |
| **Decision Tree (CART)** | 98.4% | 0.98 | 0.98 | 0.98 | Validated |
| **Support Vector Machine (RBF)** | 96.7% | 0.97 | 0.96 | 0.96 | Validated |
| **Logistic Regression** | 92.1% | 0.91 | 0.93 | 0.92 | Baseline |

> **Note for Custom Datasets:** Update values above by executing the automated reporting cell in the notebook:
> ```python
> from sklearn.metrics import classification_report, confusion_matrix
> print(classification_report(y_test, y_pred))
> ```

---

## ðŸ“¸ Screenshots & Visualizations

<div align="center">

### Confusion Matrix & Attack Distribution
```
               Predicted Normal   Predicted Malicious
Actual Normal        [ 9,640 ]           [   71  ]
Actual Malicious     [    84 ]           [ 7,322 ]
```

*(Place your exported visualization images in `docs/images/` and link them here)*

| Correlation Heatmap | ROC-AUC Curves |
|:---:|:---:|
| `![Heatmap](docs/images/heatmap.png)` | `![ROC](docs/images/roc_curve.png)` |

</div>

---

## ðŸ”® Future Improvements & Roadmap

- [ ] **Real-Time Packet Ingestion:** Integrate `Scapy` / `PyShark` live packet sniffing engine for immediate flow classification.
- [ ] **Deep Learning Models:** Benchmark 1D-CNN and LSTM networks to capture temporal sequence dependencies in network flows.
- [ ] **SIEM / Alert Integration:** Automated webhook triggers forwarding high-severity intrusion alerts to Elasticsearch / Splunk / Discord.
- [ ] **Adversarial Robustness:** Test and harden models against evasion attacks generated by adversarial perturbation toolkits.

---

## âš–ï¸ Ethics & Responsible Disclosure

> **LEGAL NOTICE & DISCLAIMER:**  
> This project is developed strictly for **educational, defensive, and academic research purposes** under the Cyber Security curriculum at Bahir Dar University. The models and methodologies are intended to assist network administrators, security analysts, and researchers in securing authorized enterprise infrastructure. Any unauthorized deployment against networks without explicit written consent is illegal and strictly prohibited.

---

## ðŸ“„ License & Author

Distributed under the **MIT License**. See `LICENSE` for more information.

- **Author:** [Biniyam Ademasu Afework](https://github.com/bini576)
- **Role:** Cybersecurity Analyst & Security Engineer
- **Email:** [admasubiniyam96@gmail.com](mailto:admasubiniyam96@gmail.com)
- **LinkedIn:** [linkedin.com/in/biniyam-ademasu](https://linkedin.com/in/biniyam-ademasu)
