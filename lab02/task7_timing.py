import time
import numpy as np
from task1_dataset import load_and_preprocess_data
from task2_network_design import initialize_parameters
from task3_forward_pass import forward_propagation
from task5_backpropagation import backward_propagation

def time_forward_and_backward(params, X_batch, Y_batch, runs=100):
    fwd_times = []
    bwd_times = []

    # Warmup
    for _ in range(10):
        preds, cache = forward_propagation(X_batch, params)
        _ = backward_propagation(cache, Y_batch, params)

    # 100 Benchmark Runs
    for _ in range(runs):
        t0 = time.perf_counter()
        preds, cache = forward_propagation(X_batch, params)
        t1 = time.perf_counter()
        _ = backward_propagation(cache, Y_batch, params)
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1e6)  # microseconds
        bwd_times.append((t2 - t1) * 1e6)

    med_fwd = np.median(fwd_times)
    med_bwd = np.median(bwd_times)

    print("=== Task 7: Timing Breakdown (Median of 100 runs) ===")
    print(f"Forward Pass:  {med_fwd:.2f} µs")
    print(f"Backward Pass: {med_bwd:.2f} µs")
    print(f"Ratio (Backward / Forward): {med_bwd / med_fwd:.2f}x")

    print("\n--- Theoretical Explanation (Matrix Multiplications) ---")
    print("The backward pass dominates because each fully connected layer requires TWO matrix multiplications:")
    print("  1. dZ * W.T (to backpropagate the error to the preceding layer)")
    print("  2. A.T * dZ (to compute parameter gradients for W)")
    print("In contrast, the forward pass requires only ONE matrix multiplication per layer (A * W).")

if __name__ == '__main__':
    np.random.seed(42)
    X_train, _, Y_train, _ = load_and_preprocess_data()
    params = initialize_parameters(hidden_dim=64)
    time_forward_and_backward(params, X_train[:32], Y_train[:32])