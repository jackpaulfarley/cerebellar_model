
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt


class ReLURidge(nn.Module):
    def __init__(self, n_features, ridge_lambda=1.0, lr=1e-2):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)
        self.ridge_lambda = ridge_lambda
        self.optimizer = optim.Adam(
            self.parameters(), lr=lr, weight_decay=self.ridge_lambda
        )
        self.loss_fn = nn.MSELoss()
        self.loss_history = []

    def forward(self, x):
        return torch.relu(self.linear(x))

    def fit(self, X_train, y_train, epochs=1000):
        self.loss_history.clear()
        for _ in trange(epochs, desc="Training"):
            self.train()
            self.optimizer.zero_grad()
            preds = self.forward(X_train)
            mse_loss = self.loss_fn(preds, y_train)

            # Don't compute L2 on the bias
            weights = self.linear.weight.view(-1)
            ridge_penalty = self.ridge_lambda * torch.sum(weights[:-1] ** 2)

            loss = mse_loss + ridge_penalty
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(mse_loss.item())

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def evaluate(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_test)
            mse = self.loss_fn(preds, y_test)
            return mse.item()

    def plot_loss(self):
        if not self.loss_history:
            print("No training history to plot.")
            return
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Loss During Training")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_weights(self, feature_names=None):
        with torch.no_grad():
            weights = self.linear.weight.view(-1).cpu().numpy()
        if feature_names:
            for name, w in zip(feature_names, weights):
                print(f"{name}: {w:.4f}")
        return weights