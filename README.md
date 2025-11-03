**💸 Financial Risk Prediction Engine**

**AI-Powered Credit Risk Assessment with Explainable Insights**

**🚀 Overview**

The Financial Risk Prediction Engine is a machine-learning pipeline designed to predict the likelihood of customer default using credit card transaction data.
It integrates XGBoost for high-accuracy classification and Explainable AI (XAI) tools like SHAP for model interpretability, helping financial institutions make transparent and data-driven credit decisions.

**🧠 Key Features**

**🏦 Credit Risk Prediction:** Identifies potential loan or credit card defaults.

**📊 Model Evaluation Dashboard:** Accuracy, F1-Score, ROC-AUC, and feature importance visualizations.

**🔍 Explainable AI Integration:** Generates SHAP summary plots for explainable model reasoning.

**⚙️ Modular Pipeline:** Clean structure for data preprocessing, training, and evaluation.

**🧰 Command-Line Execution:** Reproducible experiments using CLI arguments.

**📁 Project Structure**
financial-risk-engine/
│
├── data/                  # Raw and processed datasets (excluded from repo)
├── models/                # Trained model artifacts (.pkl)
├── notebooks/             # Jupyter experiments (optional)
├── reports/
│   └── figures/           # Evaluation plots (ROC, SHAP, etc.)
├── src/
│   ├── data_prep.py       # Data preprocessing script
│   ├── train.py           # Model training pipeline
│   ├── evaluate.py        # Model evaluation & SHAP explainability
│   ├── features.py        # Feature engineering module
│   └── api.py             # (Future) REST API integration
│
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored system & data files
└── README.md              # Project documentation


**📂 Dataset Source**

The dataset used in this project is sourced from Kaggle’s Credit Card Fraud Detection Dataset. It contains anonymized transaction features and a binary Class label indicating whether a transaction is fraudulent or legitimate.

**To download the dataset, follow these steps:**

-Log in to your Kaggle account.

-Visit the dataset page linked above.

-Click “Download” and save the file creditcard.csv inside your project’s directory (**financial-risk-engine\data\raw**).

**Alternatively, you can download it directly using the Kaggle API:**
'''''
!kaggle datasets download -d mlg-ulb/creditcardfraud
!unzip creditcardfraud.zip -d data/

'''''

Ensure your Kaggle API key (kaggle.json) is configured in your environment for seamless access.


**🧩 Tech Stack**
Category	Tools/Frameworks
Programming Language	Python 3.10+
ML Framework	XGBoost, Scikit-learn
Explainability	SHAP
Visualization	Matplotlib, Seaborn
Data Handling	NumPy, Pandas
Deployment (Planned)	FastAPI / Streamlit

**⚙️ Setup Instructions**

**1️⃣ Clone the Repository**
git clone https://github.com/KomaliG7/Financial-Risk-Prediction-Engine-.git
cd Financial-Risk-Prediction-Engine-

**2️⃣ Create Virtual Environment**
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # macOS/Linux

**3️⃣ Install Dependencies**
pip install -r requirements.txt

**4️⃣ Train the Model**
python src/train.py --processed data/processed/train.npz --out models/xgb_baseline.pkl

**5️⃣ Evaluate the Model**
python src/evaluate.py --model models/xgb_baseline.pkl --test data/processed/test.npz

**📈 Model Performance**
Metric	Score
Accuracy	0.9996
Precision	0.9405
Recall	0.8061
F1-Score	0.8681
ROC-AUC	0.9789


**📊 Visual Results**
Figure	Description

	Model’s ROC-AUC Curve

	Top features influencing predictions

	SHAP-based explainability overview

  
**🧩 Future Scope**

Integrate real-time API using FastAPI/Flask

Build Streamlit dashboard for live financial risk visualization

Extend to multi-class credit product analysis

Deploy to AWS or Render with CI/CD pipelines


**🧾 License**

This project is licensed under the MIT License — feel free to use, modify, and distribute it.


**👩‍💻 Author**
Komali G
