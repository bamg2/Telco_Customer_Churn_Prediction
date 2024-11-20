from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import encode_features
from src.model_training import train_models
from src.visualisation import plot_distribution, plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load the dataset
df = load_data('data/Telco_Customer_Churn.csv')

# Preprocess the data
df = preprocess_data(df)

# Feature engineering
df = encode_features(df)

# Split features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Define models
models = [
    ('Random Forest', RandomForestClassifier(random_state=42), {'model__n_estimators': [50, 100, 200]}),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42), {'model__n_estimators': [50, 100, 200]}),
    ('Logistic Regression', LogisticRegression(random_state=42), {'model__C': [0.1, 1, 10]}),
    ('SVM', SVC(random_state=42), {'model__C': [0.1, 1, 10]}),
    ('XGBoost', XGBClassifier(random_state=42), {'model__n_estimators': [50, 100, 200]})
]

# Train models
best_model, best_accuracy, model_scores = train_models(X, y, models)

print(f'Best Model: {best_model} with Accuracy: {best_accuracy}')
