import torch
from .models import get_optimizer
from tqdm import trange


class Flame:
    def __init__(self, model, optimizer=None, loss=None, device=None) -> None:
        self.model = model
        self.optimizer = optimizer or get_optimizer(model, lr=1e-3)
        self.loss = loss or torch.nn.MSELoss()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def train_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y, epochs=1, track_loss=0, progress=False):
        self.model.train()
        pbar = trange(epochs) if progress else range(epochs)
        pbar_msg = pbar.set_description if progress else lambda _: None
        for epoch in pbar:
            loss = self.train_step(X, y)
            pbar_msg(f"loss: {loss}")
            if track_loss and epoch % track_loss == 0:
                self.track_loss(epoch, loss)
        return loss

    def track_loss(self, epoch, loss):
        if not hasattr(self, "history"):
            self.history = {"epoch": [], "loss": []}
        self.history["epoch"].append(epoch)
        self.history["loss"].append(loss)

    def get_acc(self, X, y):
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_hat = self.model(X)
            acc = (y_hat.argmax(dim=1) == y).float().mean().item()
        return acc
        