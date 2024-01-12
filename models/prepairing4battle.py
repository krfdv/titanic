import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



df = pd.read_csv('../data/raw/train.csv')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

df = df.drop(['Sex_0', 'Embarked_0.0'], axis=1)

df.describe()
df.info()

X = df.drop('Survived', axis=1)
y = df['Survived']

X['Sex_1'] = X['Sex_1'].astype(int)
X['Embarked_1.0'] = X['Embarked_1.0'].astype(int)
X['Embarked_2.0'] = X['Embarked_2.0'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for some models like SVM and Neural Network)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# Evaluate models using cross-validation
results = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[model_name] = scores

# Print average accuracy for each model
for model_name, scores in results.items():
    print(f'{model_name}: Mean Accuracy = {np.mean(scores):.2f}, Std Deviation = {np.std(scores):.2f}')

# Train and evaluate the best model on the test set
best_model_name = max(results, key=lambda k: np.mean(results[k]))
best_model = models[best_model_name]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f'\nBest Model: {best_model_name}')
print(f'Test Accuracy: {test_accuracy:.2f}')


# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'Random Forest': {'n_estimators': [50, 100, 150, 200], 'max_depth': [None, 10, 20, 30]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150, 200], 'max_depth': [3, 4, 5]},
    'Support Vector Machine': {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Neural Network': {'hidden_layer_sizes': [(32,), (64,), (32, 16)], 'alpha': [0.0001, 0.001, 0.01]}
}

# Perform hyperparameter tuning for each model
best_models = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    print(f'Best Parameters for {model_name}: {grid_search.best_params_}')

# Evaluate the best-tuned model on the test set
best_test_results = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    best_test_results[model_name] = test_accuracy

# Print test accuracy for the best-tuned models
print('\nTest Accuracy for Best-Tuned Models:')
for model_name, test_accuracy in best_test_results.items():
    print(f'{model_name}: {test_accuracy:.2f}')
    
    
    
# Initialize Logistic Regression with L2 (Ridge) regularization
logistic_reg = LogisticRegression(penalty='l2', C=0.01)

# Train and evaluate the model as before
logistic_reg.fit(X_train, y_train)
y_pred = logistic_reg.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression (Regularized): Test Accuracy: {test_accuracy:.2f}')

# Initialize Random Forest with limited depth and fewer estimators
random_forest = RandomForestClassifier(n_estimators=100, max_depth=10)

# Train and evaluate the model as before
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest (Regularized): Test Accuracy: {test_accuracy:.2f}')

# Initialize Gradient Boosting with limited depth and fewer estimators
gradient_boosting = GradientBoostingClassifier(n_estimators=50, max_depth=3)

# Train and evaluate the model as before
gradient_boosting.fit(X_train, y_train)
y_pred = gradient_boosting.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Gradient Boosting (Regularized): Test Accuracy: {test_accuracy:.2f}')


# Initialize Support Vector Machine with different C values
svm = SVC(C=1.0, kernel='rbf')

# Train and evaluate the model as before
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Support Vector Machine (Regularized): Test Accuracy: {test_accuracy:.2f}')



# Initialize Neural Network with dropout layers
mlp = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, max_iter=1000, random_state=42, solver='adam', 
                    activation='relu', learning_rate_init=0.001, learning_rate='adaptive', early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=10, verbose=True)

# Train and evaluate the model as before
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Neural Network (Regularized): Test Accuracy: {test_accuracy:.2f}')


# Initialize additional models
models['Gaussian Naive Bayes'] = GaussianNB()
models['K-Nearest Neighbors'] = KNeighborsClassifier(n_neighbors=5)
models['Decision Tree'] = DecisionTreeClassifier()

# Evaluate additional models using cross-validation
for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[model_name] = scores

# Print average accuracy for additional models
for model_name, scores in results.items():
    print(f'{model_name}: Mean Accuracy = {np.mean(scores):.2f}, Std Deviation = {np.std(scores):.2f}')

# Train and evaluate the best model on the test set (including the new models)
best_model_name = max(results, key=lambda k: np.mean(results[k]))
best_model = models[best_model_name]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f'\nBest Model: {best_model_name}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# Ensemble using Voting Classifier
ensemble_model = VotingClassifier(estimators=[(model_name, model) for model_name, model in models.items()], voting='hard')
ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f'Ensemble Model (Voting Classifier): Test Accuracy: {test_accuracy:.2f}')

test = pd.read_csv('../data/raw/test.csv')
def preprocess_data(df):
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
  
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])
    df = df.drop(["Sex_0", "Embarked_0"], axis=1)
    
    # Ensure that the 'Sex_1', 'Embarked_1.0', and 'Embarked_2.0' columns are present
    if 'Sex_1' not in df.columns:
        df['Sex_1'] = 0
    if 'Embarked_1.0' not in df.columns:
        df['Embarked_1.0'] = 0
    if 'Embarked_2.0' not in df.columns:
        df['Embarked_2.0'] = 0
    
    return df
test = preprocess_data(test)
test["Survived"] = ensemble_model.predict(test)
test.to_csv('submission.csv', index=False)
