import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target

    num_classes = 10
    Y = np.zeros((y.size, num_classes))
    Y[np.arange(y.size), y] = 1.0

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.30, random_state=42, stratify=y
    )
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
    print("=== Task 1: Dataset Preparation Complete ===")
    print(f"X_train shape: {X_train.shape} | Y_train shape: {Y_train.shape}")
    print(f"X_test shape:  {X_test.shape}  | Y_test shape:  {Y_test.shape}")
    print(f"Input range: [{X_train.min():.2f}, {X_train.max():.2f}]")