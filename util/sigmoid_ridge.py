import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class SigReg(nn.Module):
    def __init__(self, n_features, ridge_lambda=1.0, lr=1e-2, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.C = nn.Parameter(torch.randn(n_features, device=self.device))
        self.A = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.B = nn.Parameter(torch.tensor(1.0, device=self.device))

        self.ridge_lambda = ridge_lambda
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.loss_history = []

    def forward(self, z):
        dot = torch.matmul(z, self.C)
        sigmoid = 1.0 / (1.0 + torch.exp(dot))
        return self.A + self.B * sigmoid

    def fit(self, X_train, y_train, epochs=1000, batch_size=2048):
        self.to(self.device)
        X_train = X_train.to(self.device)
        y_train = y_train.view(-1).to(self.device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        self.loss_history.clear()

        for _ in trange(epochs, desc="Training"):
            epoch_loss = 0
            for xb, yb in train_loader:
                self.train()
                self.optimizer.zero_grad()
                preds = self.forward(xb).view(-1)
                mse_loss = self.loss_fn(preds, yb)
                ridge_penalty = self.ridge_lambda * torch.sum(self.C ** 2)
                loss = mse_loss + ridge_penalty
                loss.backward()
                self.optimizer.step()
                epoch_loss += mse_loss.item() * len(xb)
            self.loss_history.append(epoch_loss / len(X_train))

    def evaluate(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.view(-1).to(self.device)
            preds = self.forward(X_test).view(-1)
            mse = self.loss_fn(preds, y_test)
            return mse.item()

    def plot_loss(self):
        if not self.loss_history:
            print("No training history to plot.")
            return
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss (MSE)")
        plt.title("Loss During Training")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_parameters(self, feature_names=None):
        with torch.no_grad():
            C_vals = self.C.cpu().numpy()
            A_val = self.A.item()
            B_val = self.B.item()
        if feature_names:
            print("C (weights):")
            for name, w in zip(feature_names, C_vals):
                print(f"  {name}: {w:.4f}")
        print(f"A (offset): {A_val:.4f}")
        print(f"B (scale): {B_val:.4f}")
        return C_vals, A_val, B_val