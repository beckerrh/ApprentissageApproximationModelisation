import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# -------- Define Neural Network (same architecture style as JAX version) ------
class PINN(nn.Module):
    def __init__(self, layers, t0, t1):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        for i, layer in enumerate(self.layers):
            in_dim = layer.weight.size(1)

            # He init (like jnp.sqrt(2/in_dim))
            nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh')

            if i == 0:
                # Spread biases across training domain
                out_dim = layer.bias.size(0)
                b = torch.linspace(t0, t1, out_dim)
                layer.bias.data = -b
            else:
                nn.init.zeros_(layer.bias)


    def forward(self, t):
        x = t
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


# --------------------------- Physics Loss -----------------------------------
class Application:
    def __init__(self, lam=1.0):
        self.t0, self.t1 = 0, 3
        self.u0 = 1.0
        self.lam = lam

    def u_true(self, t):
        return torch.exp(self.lam * t)

    def residual(self, model, t):
        t.requires_grad_(True)
        u = model(t)
        du_dt = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        return du_dt - self.lam * u

    def loss(self, model, t_colloc):
        res = self.residual(model, t_colloc)
        ode_loss = torch.mean(res**2)

        u0_pred = model(torch.tensor([[self.t0]], dtype=torch.float32))
        bc_loss = (u0_pred - self.u0)**2

        return ode_loss + bc_loss



def compute_error(app, machine, n_points=200):
    t = torch.linspace(app.t0, app.t1, n_points).view(-1, 1)  # shape (200,1)
    with torch.no_grad():
        u_pred = machine(t).numpy()
        u_true = app.u_true(t).numpy()
    return np.linalg.norm(u_true - u_pred, ord=2) / np.linalg.norm(u_true, ord=2)

# ---------------------------- Training --------------------------------------
def train(model, app, t_colloc, epochs=2000, lr=1e-3, print_every=200):
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)

    def closure():
        optimizer.zero_grad()
        loss = app.loss(model, t_colloc)
        loss.backward()
        return loss

    for epoch in range(epochs):
        loss = optimizer.step(closure)
        if epoch % print_every == 0:
            print(f"Epoch {epoch:5d}, Loss = {loss.item():.3e}")


# -------------------------------- Main --------------------------------------
app = Application(lam=1.0)
layers = [1, 8, 8, 1]
model = PINN(layers, app.t0, app.t1)

t_plot = torch.linspace(app.t0, app.t1, 200).view(-1,1)
n_colloc = 40
t_colloc = torch.linspace(app.t0, app.t1, n_colloc).view(-1,1)

train(model, app, t_colloc, epochs=400, print_every=20, lr=0.1)

# Visualization
u_pred = model(t_plot).detach()
u_true = app.u_true(t_plot).detach()

# plt.figure(figsize=(6,4))
# plt.plot(t_plot, u_pred, label="Approximation")
# plt.plot(t_plot, u_true, "--", label="Exact")
# plt.scatter(t_colloc.detach(), torch.zeros_like(t_colloc).detach(), c='r', marker='x', label='Collocation')
# plt.title("PINN approximation of exp(t)")
# plt.legend()
# plt.grid(True)
# plt.show()
print(f"Relative L2 error: {compute_error(app, model):.3e}")
