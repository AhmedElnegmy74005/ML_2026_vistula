import numpy as np
from task1_dataset import load_and_preprocess_data
from task2_network_design import initialize_parameters
from task3_forward_pass import forward_propagation

def compute_cross_entropy_loss(Y_true, Y_pred):
    m = Y_true.shape[0]
    eps = 1e-15
    loss = -np.sum(Y_true * np.log(np.clip(Y_pred, eps, 1.0))) / m
    return loss

if __name__ == '__main__':
    np.random.seed(42)
    X_train, _, Y_train, _ = load_and_preprocess_data()
    params = initialize_parameters(hidden_dim=16)

    # Sanity check on batch of 32
    probs, _ = forward_propagation(X_train[:32], params)
    loss = compute_cross_entropy_loss(Y_train[:32], probs)

    expected = np.log(10)
    print("=== Task 4: Cross-Entropy Sanity Check ===")
    print(f"Computed untrained loss: {loss:.4f}")
    print(f"Expected reference loss: {expected:.4f} (ln(10) = 2.3026)")
    print(f"Difference: {abs(loss - expected):.6f}")