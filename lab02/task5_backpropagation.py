import numpy as np
from task1_dataset import load_and_preprocess_data
from task2_network_design import initialize_parameters
from task3_forward_pass import forward_propagation
from task4_cross_entropy import compute_cross_entropy_loss

def backward_propagation(cache, Y_true, params):
    m = Y_true.shape[0]
    grads = {}

    dZ3 = cache['A3'] - Y_true
    grads['W3'] = np.dot(cache['A2'].T, dZ3) / m
    grads['b3'] = np.sum(dZ3, axis=0) / m

    # Layer 2
    dA2 = np.dot(dZ3, params['W3'].T)
    dZ2 = dA2 * (cache['Z2'] > 0).astype(float)
    grads['W2'] = np.dot(cache['A1'].T, dZ2) / m
    grads['b2'] = np.sum(dZ2, axis=0) / m

    # Layer 1
    dA1 = np.dot(dZ2, params['W2'].T)
    dZ1 = dA1 * (cache['Z1'] > 0).astype(float)
    grads['W1'] = np.dot(cache['A0'].T, dZ1) / m
    grads['b1'] = np.sum(dZ1, axis=0) / m

    return grads

def run_gradient_check(X_sample, Y_sample, params, epsilon=1e-5):
    _, cache = forward_propagation(X_sample, params)
    analytical_grads = backward_propagation(cache, Y_sample, params)

    passed = 0
    total = 10
    keys = ['W1', 'W2', 'W3']

    print("=== Task 5: Numerical Gradient Checking ===")
    for i in range(total):
        k = np.random.choice(keys)
        idx = tuple(np.random.randint(0, d) for d in params[k].shape)
        orig = params[k][idx]

        # +eps
        params[k][idx] = orig + epsilon
        p_plus, _ = forward_propagation(X_sample, params)
        l_plus = compute_cross_entropy_loss(Y_sample, p_plus)

        # -eps
        params[k][idx] = orig - epsilon
        p_minus, _ = forward_propagation(X_sample, params)
        l_minus = compute_cross_entropy_loss(Y_sample, p_minus)

        params[k][idx] = orig
        num_grad = (l_plus - l_minus) / (2.0 * epsilon)
        ana_grad = analytical_grads[k][idx]

        diff = abs(num_grad - ana_grad)
        rel_err = diff if (abs(num_grad) < 1e-6 and abs(ana_grad) < 1e-6) else diff / (abs(num_grad) + abs(ana_grad))
        is_ok = rel_err < 1e-6
        if is_ok:
            passed += 1
        print(f"[{i+1}/10] {k}{idx} | Analytic: {ana_grad:.8e} | Numeric: {num_grad:.8e} | Rel Err: {rel_err:.2e} | Passed: {is_ok}")

    print(f"\nResult: {passed}/{total} passed (Required: >= 9/10)")
    assert passed >= 9, "Gradient check failed!"

if __name__ == '__main__':
    np.random.seed(42)
    X_train, _, Y_train, _ = load_and_preprocess_data()
    params = initialize_parameters(hidden_dim=16)
    run_gradient_check(X_train[:32], Y_train[:32], params)