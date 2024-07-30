import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv('../data/iris.csv', index=False)
    # Load dataset
    data = pd.read_csv('../data/iris.csv')
    X = data.drop(columns=['target'])
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(model_name, model, params):
    X_train, X_test, y_train, y_test = load_data()

    # Train model
    model.set_params(**params)
    model.fit(X_train, y_train)

    # Predict and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def log_experiment(model_name, params, accuracy, model):
    with mlflow.start_run():
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, model_name)


if __name__ == "__main__":
    models = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVC": SVC()
    }

    params_grid = {
        "RandomForest": [
            {"n_estimators": 10, "max_depth": 3},
            {"n_estimators": 50, "max_depth": 5},
            {"n_estimators": 100, "max_depth": 7}
        ],
        "GradientBoosting": [
            {"n_estimators": 10, "max_depth": 3},
            {"n_estimators": 50, "max_depth": 5},
            {"n_estimators": 100, "max_depth": 7}
        ],
        "SVC": [
            {"C": 0.1, "kernel": "linear"},
            {"C": 1, "kernel": "poly"},
            {"C": 10, "kernel": "rbf"}
        ]
    }

    for model_name, model in models.items():
        for params in params_grid[model_name]:
            model, accuracy = train_model(model_name, model, params)
            log_experiment(model_name, params, accuracy, model)
            print(f"Run with {model_name} and params={params} has accuracy={accuracy}")
