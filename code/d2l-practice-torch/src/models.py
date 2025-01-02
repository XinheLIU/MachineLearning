from d2l import torch as d2l
import torch
from torch import nn

class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def prepare_batch(self, batch):
        return batch

    def fit_epoch(self):
        """
        Train the model for one epoch and validate it.
        """
        # Set the model to training mode
        self.model.train()
        # Iterate over the training data loader
        for batch in self.train_dataloader:
            # Perform a training step and get the loss
            loss = self.model.training_step(self.prepare_batch(batch))
            # Clear the gradients of the optimizer
            self.optim.zero_grad()
            # Perform backpropagation without gradient calculation
            with torch.no_grad():
                loss.backward()
                # If gradient clipping is enabled, clip the gradients
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                # Update the parameters of the optimizer
                self.optim.step()
            # Increment the training batch index
            self.train_batch_idx += 1
        # If there is no validation data loader, return
        if self.val_dataloader is None:
            return
        # Set the model to evaluation mode
        self.model.eval()
        # Iterate over the validation data loader
        for batch in self.val_dataloader:
            # Perform a validation step without gradient calculation
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            # Increment the validation batch index
            self.val_batch_idx += 1

# d2l.Module
# def apply_init(self, inputs, init=None):
#     self.forward(*inputs)
#     if init is not None:
#         self.net.apply(init)

class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()
    def configure_optimizers(self):
        return torch.optim.SGD([self.w, self.b], self.lr)
    
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
    
    def forward(self, X):
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

def l2_penalty(w):
    return (w ** 2).sum() / 2

class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
    
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
    
        Returns:
            torch.optim.SGD: The configured optimizer.
        """
        # Return a stochastic gradient descent optimizer with weight decay
        return torch.optim.SGD([
            # Apply weight decay to the weights of the network
            {'params': self.net.weight, 'weight_decay': self.wd},
            # Do not apply weight decay to the bias of the network
            {'params': self.net.bias}], lr=self.lr)