import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# -------- Define Neural Network ------
class MachineODE(nn.Module):
    def __init__(self, layers, app):
        super().__init__()
        self.layers = nn.ModuleList()
        self.app = app

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        for i, layer in enumerate(self.layers):
            in_dim = layer.weight.size(1)
            out_dim = layer.bias.size(0)
            if i == 0:
                layer.bias.data = -torch.linspace(self.app.t0, self.app.t1, out_dim)
                nn.init.constant_(layer.weight, 1.0)
            else:
                nn.init.zeros_(layer.bias)
                if in_dim == out_dim:
                    nn.init.eye_(layer.weight)
                else:
                    nn.init.constant_(layer.weight, 0.0)


    def forward(self, t):
        x = t
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)
    def residual(self, t):
        t.requires_grad_(True)
        u = self.forward(t)
        du_dt = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        return du_dt - self.app.f(u)

    def loss(self, t_colloc):
        res = self.residual(t_colloc)
        ode_loss = torch.mean(res**2)

        u0_pred = self.forward(torch.tensor([[self.app.t0]]))
        bc_loss = (u0_pred - self.app.u0)**2

        return ode_loss + bc_loss




# --------------------------- Error Computation ------------------------------

def compute_error(app, machine, n_points=2000):
    t = torch.linspace(app.t0, app.t1, n_points).view(-1, 1)  # shape (200,1)
    with torch.no_grad():
        u_pred = machine(t).numpy()
        u_true = app.u_true(t).numpy()
    return np.linalg.norm(u_true - u_pred, ord=2) / np.linalg.norm(u_true, ord=2)

# ---------------------------- Training --------------------------------------
def train(machine, t_colloc, epochs=1000, lr=1.0, print_every=20, eps=1e-8):
    optimizer = optim.LBFGS(machine.parameters(), lr=lr, max_iter=1)

    def closure():
        optimizer.zero_grad()
        loss = machine.loss(t_colloc)
        loss.backward()
        return loss

    for epoch in range(epochs):
        loss = optimizer.step(closure)
        if epoch % print_every == 0:
            print(f"Epoch {epoch:5d}, Loss = {loss.item():.3e}")
            if epoch and abs(loss.item() - loss_old) < eps:
                return
            loss_old = loss.item()


# -------------------------------- Main --------------------------------------
class ApplicationExp:
    def __init__(self):
        self.t0, self.t1 = 0.0, 3.0
        self.u0 = 1.0
        self.lam = 1.0
    def u_true(self, t):
        return np.exp(self.lam*t)
    def f(self, u):
        return self.lam * u
class ApplicationSin:
    def __init__(self):
        self.t0, self.t1 = 0.0, 3.0
        self.u0 = 0.01
        self.lam = 1.2
        self.K = np.tan(0.5 * self.lam * self.u0)
    def u_true(self, t):
        return 2.0/self.lam*torch.arctan(self.K * torch.exp(self.lam * t))
    def f(self, u):
        return torch.sin(self.lam * u)

app = ApplicationSin()
layers = [1, 8, 8, 1]
torch.set_default_dtype(torch.float64)

machine = MachineODE(layers, app)

t_plot = torch.linspace(app.t0, app.t1, 200).view(-1,1)
n_colloc = 40
t_colloc = torch.linspace(app.t0, app.t1, n_colloc).view(-1,1)

train(machine, t_colloc, eps=1e-10)

# Visualization
u_pred = machine(t_plot).detach()
u_true = app.u_true(t_plot).detach()

plt.figure(figsize=(6,4))
plt.plot(t_plot, u_pred, label="Approximation")
plt.plot(t_plot, u_true, "--", label="Exact")
plt.scatter(t_colloc.detach(), torch.zeros_like(t_colloc).detach(), c='r', marker='x', label='Collocation')
plt.title("Comparison")
plt.legend()
plt.grid(True)
plt.show()
print(f"Relative L2 error: {compute_error(app, machine):.3e}")
