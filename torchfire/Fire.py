import torch
from .models import get_optimizer
from tqdm import trange
from collections import defaultdict


class Fire:
    def __init__(self, model, optimizer=None, loss="mse", device=None) -> None:
        self.model = model
        self.optimizer = optimizer or get_optimizer(model, lr=1e-3)
        if loss == "mse":
            self.loss = torch.nn.MSELoss()
        elif loss == "xent":
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"loss {loss} not supported")
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
        return loss

    def eval_step(self, x, y):
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
        return loss

    def eval(self, loader):
        agg = _Aggregator()
        for X, y in loader:
            loss_batch = self.eval_step(X, y).item()
            agg(loss_batch, size=len(X))
        return agg.value

    def fit(self, X, y=None, valloader=None, epochs=1, track_loss=0, progress=False):
        loader = X if y is None else [(X, y)]  # if y is None: assume X is a dataloader
        valloader = valloader or []
        self.model.train()
        pbar = trange(epochs) if progress else range(epochs)
        pbar_msg = pbar.set_description if progress else lambda _: None
        for epoch in pbar:
            agg = _Aggregator()
            for X, y in loader:
                loss_batch = self.train_step(X, y).item()
                loss = agg(loss_batch, size=len(X))
            pbar_msg(f"loss: {loss}")
            if track_loss and epoch % track_loss == 0:
                val_loss = self.eval(valloader)
                self.track(epoch, loss=loss, val_loss=val_loss)
        return loss

    def track(self, epoch, **kwargs):
        """track metrics in self.history"""
        if not hasattr(self, "history"):
            self.history = defaultdict(list)
        self.history["epoch"].append(epoch)
        for metric_name, metric_value in kwargs.items():
            if metric_value is not None:
                self.history[metric_name].append(metric_value)

    def get_acc(self, X, y):
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_hat = self.model(X)
            acc = (y_hat.argmax(dim=1) == y).float().mean().item()
        return acc


class _Aggregator:
    def __init__(self, agg_fn="mean") -> None:
        assert agg_fn in ["mean", "sum"], f"agg_fn {agg_fn} not supported"
        self.value = None
        self.size = 0
        self.agg_fn = agg_fn

    def __call__(self, new_value, size=1):
        if self.value is None:
            self.value = new_value
            self.size = size
        if self.agg_fn == "mean":
            self.value = (self.value * self.size + new_value * size) / (
                self.size + size
            )
        elif self.agg_fn == "sum":
            self.value += new_value * size
        return self.value
