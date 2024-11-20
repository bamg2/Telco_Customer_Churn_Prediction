from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(X, y, models):
    """Trains models and returns the best model."""
    best_model = None
    best_accuracy = 0.0
    model_scores = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model, param_grid in models:
        pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', model)])

        if param_grid:
            grid_search = GridSearchCV(pipeline, param_grid, cv=3)
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        model_scores.append({'Model': name, 'Accuracy': accuracy})

        if accuracy > best_accuracy:
            best_model = pipeline
            best_accuracy = accuracy

    return best_model, best_accuracy, model_scores
