import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os


# Function for data preprocessing
def preprocess_data(df):
    df[cabin]
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])
    df = df.drop(["Sex_male", "Embarked_Q"], axis=1)
    return df


# Function for model evaluation
def evaluate_model(model: VotingClassifier, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy


# Function for hyperparameter tuning
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params


# Function to run the entire pipeline
def main():
    # Load and preprocess data
    df = pd.read_csv("data/raw/train.csv")
    df = preprocess_data(df)

    # Split data into features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=41
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Support Vector Machine": SVC(),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
    }

    # Evaluate models using cross-validation

    # Train and evaluate the best model on the test set

    # Define hyperparameter grids for each model
    param_grids = {
        "Logistic Regression": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        "Random Forest": {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [None, 10, 20, 30],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 4, 5],
        },
        "Support Vector Machine": {
            "C": [0.01, 0.1, 1, 10],
            "kernel": ["linear", "rbf"],
        },
        "Neural Network": {
            "hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "alpha": [0.0001, 0.001, 0.01],
        },
        "Gaussian Naive Bayes": {},  # No hyperparameters to tune for Gaussian Naive Bayes
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Decision Tree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    }

    # Perform hyperparameter tuning for each model
    best_models = {}
    for model_name, model in models.items():
        param_grid = param_grids.get(
            model_name, {}
        )  # Get hyperparameter grid for the model
        best_model, best_params = tune_hyperparameters(
            model, param_grid, X_train, y_train
        )
        best_models[model_name] = best_model
        print(f"Best Parameters for {model_name}: {best_params}")

    # Evaluate the best-tuned model on the test set
    # best_test_results = {}
    # for model_name, model in best_models.items():
    #     test_accuracy = evaluate_model(model, X_train, y_train, X_test, y_test)
    #     best_test_results[model_name] = test_accuracy
    #     print(f"{model_name} (Tuned): Test Accuracy: {test_accuracy:.2f}")
    results = {}
    for model_name, model in best_models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        results[model_name] = scores
        print(
            f"{model_name}: Mean Accuracy = {np.mean(scores):.2f}, Std Deviation = {np.std(scores):.2f}"
        )

    best_model_name = max(results, key=lambda k: np.mean(results[k]))
    best_model = best_models[best_model_name]
    test_accuracy = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # # Ensemble using Voting Classifier
    # ensemble_model = VotingClassifier(
    #     estimators=[(model_name, model) for model_name, model in best_models.items()],
    #     voting="hard",
    # )
    # ensemble_accuracy = evaluate_model(ensemble_model, X_train, y_train, X_test, y_test)
    # print(f"Ensemble Model (Voting Classifier): Test Accuracy: {ensemble_accuracy:.2f}")


if __name__ == "__main__":
    main()
