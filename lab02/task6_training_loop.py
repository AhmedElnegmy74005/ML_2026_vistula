import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from task1_dataset import load_and_preprocess_data
from task2_network_design import initialize_parameters
from task3_forward_pass import forward_propagation
from task4_cross_entropy import compute_cross_entropy_loss
from task5_backpropagation import backward_propagation

def train_model(X_train, Y_train, X_test, Y_test, hidden_dim, lr, epochs=50, batch_size=32):
    params = initialize_parameters(hidden_dim=hidden_dim)
    m = X_train.shape[0]
    loss_history = []

    for ep in range(epochs):
        indices = np.random.permutation(m)
        X_shuff = X_train[indices]
        Y_shuff = Y_train[indices]
        epoch_loss = 0.0
        n_batches = int(np.ceil(m / batch_size))

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, m)
            X_b, Y_b = X_shuff[start:end], Y_shuff[start:end]

            preds, cache = forward_propagation(X_b, params)
            loss = compute_cross_entropy_loss(Y_b, preds)
            epoch_loss += loss * (end - start)

            grads = backward_propagation(cache, Y_b, params)
            for k in params:
                params[k] -= lr * grads[k]

        epoch_loss /= m
        loss_history.append(epoch_loss)

    test_preds, _ = forward_propagation(X_test, params)
    acc = np.mean(np.argmax(test_preds, axis=1) == np.argmax(Y_test, axis=1))
    return loss_history, acc, params

if __name__ == '__main__':
    np.random.seed(42)
    os.makedirs('results', exist_ok=True)
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data()

    widths = [16, 64]
    lrs = [0.001, 0.01, 0.1]
    epochs = 50

    summary_data = []
    loss_curves_data = {}

    print("=== Task 6: Training Experiments ===")
    for w in widths:
        for lr in lrs:
            run_key = f"W={w}_lr={lr}"
            losses, acc, _ = train_model(X_train, Y_train, X_test, Y_test, hidden_dim=w, lr=lr, epochs=epochs)
            loss_curves_data[run_key] = losses
            summary_data.append({
                'hidden_width': w,
                'learning_rate': lr,
                'final_loss': round(losses[-1], 6),
                'test_accuracy_pct': round(acc * 100, 2)
            })
            print(f"Trained W={w:2d}, lr={lr:<5} | Final Loss: {losses[-1]:.4f} | Test Accuracy: {acc * 100:.2f}%")

    # 1. Individual Plot for W = 16
    plt.figure(figsize=(8, 5))
    for lr in lrs:
        acc = [row['test_accuracy_pct'] for row in summary_data if row['hidden_width'] == 16 and row['learning_rate'] == lr][0]
        plt.plot(loss_curves_data[f"W=16_lr={lr}"], label=f"lr={lr} (Acc: {acc}%)")
    plt.title("Cross-Entropy Loss (Hidden Width W = 16)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_curves_W16.png", dpi=300)
    plt.close()
    print("\nSaved separate plot: results/loss_curves_W16.png")

    # 2. Individual Plot for W = 64
    plt.figure(figsize=(8, 5))
    for lr in lrs:
        acc = [row['test_accuracy_pct'] for row in summary_data if row['hidden_width'] == 64 and row['learning_rate'] == lr][0]
        plt.plot(loss_curves_data[f"W=64_lr={lr}"], label=f"lr={lr} (Acc: {acc}%)")
    plt.title("Cross-Entropy Loss (Hidden Width W = 64)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_curves_W64.png", dpi=300)
    plt.close()
    print("Saved separate plot: results/loss_curves_W64.png")

    # 3. Side-by-Side Plot for results/loss_curves.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, w in enumerate(widths):
        ax = axes[idx]
        for lr in lrs:
            acc = [row['test_accuracy_pct'] for row in summary_data if row['hidden_width'] == w and row['learning_rate'] == lr][0]
            ax.plot(loss_curves_data[f"W={w}_lr={lr}"], label=f"lr={lr} (Acc: {acc}%)")
        ax.set_title(f"Hidden Width W = {w}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    plt.suptitle("Cross-Entropy Loss Curves across Learning Rates & Widths", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/loss_curves.png", dpi=300)
    plt.close()
    print("Saved side-by-side plot: results/loss_curves.png")

    # Save Summary CSV
    with open("results/experiment_summary.csv", mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['hidden_width', 'learning_rate', 'final_loss', 'test_accuracy_pct'])
        writer.writeheader()
        writer.writerows(summary_data)
    print("\nSaved results/experiment_summary.csv")

    # Save Epoch Loss History CSV
    with open("results/training_loss_history.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch'] + list(loss_curves_data.keys()))
        for ep in range(epochs):
            writer.writerow([ep + 1] + [loss_curves_data[k][ep] for k in loss_curves_data])
    print("Saved results/training_loss_history.csv")