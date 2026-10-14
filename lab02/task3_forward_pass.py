import numpy as np
from task1_dataset import load_and_preprocess_data
from task2_network_design import initialize_parameters

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def forward_propagation(X, params):
    cache = {'A0': X}
    # Layer 1
    cache['Z1'] = np.dot(X, params['W1']) + params['b1']
    cache['A1'] = relu(cache['Z1'])
    # Layer 2
    cache['Z2'] = np.dot(cache['A1'], params['W2']) + params['b2']
    cache['A2'] = relu(cache['Z2'])
    # Output Layer
    cache['Z3'] = np.dot(cache['A2'], params['W3']) + params['b3']
    cache['A3'] = softmax(cache['Z3'])
    return cache['A3'], cache

if __name__ == '__main__':
    np.random.seed(42)
    X_train, _, _, _ = load_and_preprocess_data()
    params = initialize_parameters(hidden_dim=16)

    # Test with batch of 5 samples
    X_batch = X_train[:5]
    probs, cache = forward_propagation(X_batch, params)

    print("=== Task 3: Forward Pass Sanity Checks ===")
    print(f"Output shape: {probs.shape} (Expected: (5, 10))")
    row_sums = np.sum(probs, axis=1)
    print(f"Row sums: {row_sums}")
    is_valid_sum = np.allclose(row_sums, 1.0, atol=1e-6)
    print(f"Each row sums to 1 within 1e-6: {is_valid_sum}")