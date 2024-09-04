import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

def evaluate():
    # Load model
    model = joblib.load('model.joblib')
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Evaluate model
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=iris.target_names)
    print("Model Evaluation:")
    print(report)

if __name__ == "__main__":
    evaluate()