import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def test():
    # Load model
    model = joblib.load('model.joblib')
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test model
    score = model.score(X_test, y_test)
    print(f"Model test score1: {score}")

if __name__ == "__main__":
    test()