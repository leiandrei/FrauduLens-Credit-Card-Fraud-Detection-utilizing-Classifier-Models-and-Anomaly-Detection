# **FrauduLENS: A Credit Card Fraud Detection using Classifier Models and Anomaly Detection**

## **Overview:**
This project aims to determine the anomalies between the fradulent and non-fraudulent credit cards within the dataset using Supervised Machine Learning Classifiers such as Logistic Regression as a baseline and linear model, and more complex models such as K-Nearest Neighbors and Reandom Forest Classifiers. Moreover, using Unsupervised Learning for Anomaly Detection using Isolation Forests.

## **Key Insights**
- Using Exploratory Data Analysis, the dataset class is highly-imbalanced garnering only 0.17% fraud transactions vs 99.83% normal transactions.
- SMOTE Sampling handles the highly-imbalanced dataset by creating a synthetic data points for the minority class.
- A Random Forest Classifier maintained a more balanced trade-off between precision and recall while maintaining high accuracy score and ROC-Area Under Curve score, even without tuned parameters.
- A more balanced precsion-recall trade-off can minimize False Positives and False Negatives in fraud detections, this ensures that true fraudulent cases can be detected without flagging them as not fraudulent, and vice versa. 

## **Project Structure**
```text
Credit Card Fraud Detection/
├── .venv/                      # Python virtual environment (ignored in git)
├── data/                       # Dataset directory (ignored in git)
│   ├── processed/              # Cleaned and preprocessed data ready for modeling
│   │   └── creditcard-processed.csv
│   └── raw/                    # Original, unmodified datasets
│       └── creditcard.csv
├── docs/                       # Project documentation and technical reports
│   └── technical_report.pdf    
├── models/                     # Serialized, deployment-ready models
│   └── tuned_rfs_model.joblib
├── notebooks/                  # Jupyter notebooks for interactive development
│   ├── exploratory_da/         # Exploratory Data Analysis
│   │   └── eda.ipynb
│   ├── model_creation/         # Baseline model training and experimentation
│   │   ├── isolation_forest.ipynb
│   │   ├── knn.ipynb
│   │   ├── logreg.ipynb
│   │   ├── random_forests.ipynb
│   │   └── README.md           # Documentation specific to baseline models
│   └── model_evaluation/       # Hyperparameter tuning and performance metrics
│       ├── README.md           # Documentation for evaluation metrics
│       ├── tuned_knn.ipynb
│       ├── tuned_log.ipynb
│       └── tuned_rfs.ipynb
├── src/                        # Reusable utility scripts and functions
│   ├── __pycache__/            
│   └── utils.py                # Core pipeline utility functions
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── config.yaml                 # Centralized configuration parameters (paths, hyperparameters)
├── LICENSE                     # Open-source license definition
├── README.md                   # Project overview and setup instructions
└── requirements.txt            # Python dependencies and library versions
```

## **Tech Stack**
- Language: Python 3.10+
- Data Processing: `numpy`, `pandas`
- Visualization: `matplotlib`, `seaborn`
- ML Modeling: `scikit-learn`, `imblearn`
- Optimization: `optuna`, `skopt`
- Environment: `venv`, `jupyter`

## **Cloning the Repository**
1. Clone the Repository:
    ```bash
    git clone https://github.com/your-username/FrauduLens-Credit-Card-Fraud-Detection-utilizing-Classifier-Models-and-Anomaly-Detection.git

    cd FrauduLens-Credit-Card-Fraud-Detection-utilizing-Classifier-Models-and-Anomaly-Detection
    ```

2. Create and activate a virual env:
    ```bash
    py -m venv .venv
    source .venv/Scripts/activate
    ```

3. Install dependencies:
    ```bash
    pip instal -r requirements.txt
    ```

4. Configure paths and parameters:
   Update `config.yaml` with any local file paths or parameters specific to your environment.

## **Dataset from Kaggle**
This dataset was sourced out from Kaggle, and contains anonymized credit card transactions labeled as genuine and fraudulent. (MLG - ULB, 2018)
```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

## **LICENSE**
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.