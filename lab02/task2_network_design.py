import numpy as np

def initialize_parameters(input_dim=64, hidden_dim=16, output_dim=10):
    # Weights sampled from N(0, 0.01), biases initialized to 0
    params = {
        'W1': np.random.normal(0.0, 0.01, (input_dim, hidden_dim)),
        'b1': np.zeros((hidden_dim,)),
        'W2': np.random.normal(0.0, 0.01, (hidden_dim, hidden_dim)),
        'b2': np.zeros((hidden_dim,)),
        'W3': np.random.normal(0.0, 0.01, (hidden_dim, output_dim)),
        'b3': np.zeros((output_dim,))
    }
    return params

if __name__ == '__main__':
    np.random.seed(42)
    for W in [16, 64]:
        params = initialize_parameters(hidden_dim=W)
        print(f"\n=== Task 2: Parameter Shapes for W = {W} ===")
        for name, tensor in params.items():
            print(f"  {name:2s}: shape = {tensor.shape}")