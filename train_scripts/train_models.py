import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

def load_preprocessed_data(filepath='data/training_data.csv'):
    df = pd.read_csv(filepath)
    X = df.drop(['placement', 'estimated_package'], axis=1)
    y_placement = df['placement']
    y_salary = df['estimated_package']
    return X, y_placement, y_salary

def load_preprocessor(filepath='models/preprocessor.joblib'):
    return joblib.load(filepath)

def train_classification_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_regression_model(X_train, y_train):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg

def save_model(model, filepath):
    joblib.dump(model, filepath)

def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/training_data.csv')
    X = df.drop(['placement', 'estimated_package'], axis=1)
    y_placement = df['placement']
    y_salary = df['estimated_package']
    
    # Load preprocessor
    preprocessor = load_preprocessor('models/preprocessor.joblib')
    
    # Preprocess features
    X_processed = preprocessor.transform(X)
    
    # Split data for classification (placement)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_processed, y_placement, test_size=0.2, random_state=42
    )
    
    # Train Logistic Regression
    clf_model = train_classification_model(X_train_clf, y_train_clf)
    clf_accuracy = evaluate_classification_model(clf_model, X_test_clf, y_test_clf)
    print(f"Logistic Regression Accuracy: {clf_accuracy:.2f}")
    
    # Train Random Forest Classifier
    rf_clf_model = train_random_forest(X_train_clf, y_train_clf)
    rf_clf_accuracy = evaluate_classification_model(rf_clf_model, X_test_clf, y_test_clf)
    print(f"Random Forest Classifier Accuracy: {rf_clf_accuracy:.2f}")
    
    # Save classification models
    save_model(clf_model, 'models/placement_model.pkl')
    save_model(rf_clf_model, 'models/rf_placement_model.pkl')
    print("Classification models saved.")
    
    # Split data for regression (salary)
    # Only consider rows where placement = 1
    regression_df = df[df['placement'] == 1]
    X_reg = regression_df.drop(['placement', 'estimated_package'], axis=1)
    y_reg = regression_df['estimated_package']
    X_reg_processed = preprocessor.transform(X_reg)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_processed, y_reg, test_size=0.2, random_state=42
    )
    
    # Train Regression Model
    reg_model = train_regression_model(X_train_reg, y_train_reg)
    reg_mse = evaluate_regression_model(reg_model, X_test_reg, y_test_reg)
    print(f"Regression Model RMSE: {reg_mse:.2f}")
    
    # Save regression model
    save_model(reg_model, 'models/salary_model.pkl')
    print("Regression model saved.")