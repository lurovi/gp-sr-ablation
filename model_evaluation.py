from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(estimator, X_train, X_test, y_train, y_test, metric="r2"):
    """
    Evaluate the trained pipeline on train and test data using the specified metric.

    Parameters:
    - estimator: The trained estimator
    - X_train, X_test: Feature matrices for training and testing
    - y_train, y_test: Target values for training and testing
    - metric: String specifying the evaluation metric ("r2", "mae", "mse", "rmse")
    """
    metrics = {
        "r2": r2_score,
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error
    }

    if metric not in metrics:
        raise ValueError(f"Unsupported metric: {metric}. Choose from {list(metrics.keys())}")

    scoring_function = metrics[metric]

    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    train_score = scoring_function(y_train, y_train_pred)
    test_score = scoring_function(y_test, y_test_pred)

    return {"train_score": train_score, "test_score": test_score, "train_pred": y_train_pred, "test_pred": y_test_pred}
