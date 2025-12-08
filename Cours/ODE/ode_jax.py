import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
jax.config.update("jax_enable_x64", True)

# Visualization
def plot_solutions(t_plot, t_colloc, u1, u2, t1='Approximation', t2='Solution', ls1='', ls2='--', title="Title"):
    plt.plot(t_plot, u1, ls1, label=t1)
    plt.plot(t_plot, u2, ls2, label=t2)
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='t_colloc')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(title)
    plt.grid(True)
    plt.show()
def plot_bases(t_plot, phi, title="Fonctions de base"):
    plt.figure(figsize=(7,3))
    plt.plot(t_plot, phi.T)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(r"$\varphi_k(t)$")
    plt.grid(True)
    plt.legend()
    plt.show()

# Apprentissage
def train(params, machine, optimizer=None, n_epochs=1000, print_every=20, learning_rate=1.0, eps=1e-8, **kwargs):
    if optimizer is None: optimizer = optax.lbfgs(learning_rate=learning_rate)
   
    # Initialize optimizer state
    opt_state = optimizer.init(params)

    loss_fn = lambda p: machine.loss(params=p, **kwargs)

    # Define one train step
    @jax.jit
    def train_step(params, opt_state, **kwargs):
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads, updates

    # Training loop
    for epoch in range(n_epochs):
        params, opt_state, loss_value, grads, updates = train_step(params, opt_state, **kwargs)
        if epoch % print_every == 0:
            print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")
            if epoch and abs(loss_value - loss_old) < eps:
                return params, opt_state
            loss_old = loss_value

    return params, opt_state

# -------------------------------------------------------------------------------- #
# Machine
class MachineODE:
    def __init__(self, layers, app):
        self.layers = layers
        self.app = app
        #key = jax.random.PRNGKey(0)
        params = []
        for l in range(1, len(layers)):
            in_dim, out_dim = layers[l-1], layers[l]
            #key, subkey = jax.random.split(key)
            #W = jax.random.normal(subkey, (out_dim, in_dim)) * jnp.sqrt(2/in_dim)
            if l==1: 
                b = -jnp.linspace(app.t0, app.t1, out_dim)
                W = jnp.ones((out_dim, in_dim))
            else: 
                b = jnp.zeros(out_dim)
                if in_dim == out_dim:
                    W = jnp.eye(in_dim)
                else:
                    W = jnp.zeros((out_dim, in_dim))
            params.append((W,b))
        self.params = params
    def basis(self, t, params=None):
        if params is None:
            params = self.params
        t = jnp.atleast_1d(t).reshape(1, -1)
        for param in params[:-1]:
            W, b = param
            t = jnp.tanh(W @ t + b[:, None])
        return t
    def forward(self, t, params=None):
        if params is None:
            B = self.basis(t)
            W, b = self.params[-1]
            return (W @ B +b).squeeze()
        B = self.basis(t, params)
        W, b = params[-1]
        return (W @ B +b).squeeze()
    def residual(self, params, t):
        dudt_scalar = jax.grad(self.forward, argnums=0)
        dudt = jax.vmap(lambda ti: dudt_scalar(ti, params))
        return dudt(t) - self.app.f(self.forward(t, params))
    def loss(self, params, t):
        ode_loss = jnp.mean(self.residual(params, t)**2)
        bc_loss = (self.forward(self.app.t0, params)-self.app.u0) ** 2 
        return ode_loss + bc_loss

def compute_error(app, machine, n_points=2000):
    t = np.linspace(app.t0, app.t1, n_points)
    u_pred = machine.forward(t)
    u_true = app.u_true(t)
    error = np.linalg.norm(u_true - u_pred, ord=2) / np.linalg.norm(u_true, ord=2)
    return error

# -------------------------------------------------------------------------------- #
class ApplicationExp:
    def __init__(self):
        self.t0, self.t1 = 0.0, 3.0
        self.u0 = 1.0
        self.lam = 1.0
    def u_true(self, t):
        return jnp.exp(self.lam*t)
    def f(self, u):
        return self.lam * u
class ApplicationSin:
    def __init__(self):
        self.t0, self.t1 = 0.0, 3.0
        self.u0 = 0.01
        self.lam = 1.2
        self.K = jnp.tan(0.5 * self.lam * self.u0)
    def u_true(self, t):
        return 2.0/self.lam*jnp.arctan(self.K * jnp.exp(self.lam * t))
    def f(self, u):
        return jnp.sin(self.lam * u)

app = ApplicationSin()
layers = [1, 8, 8, 1]
machine = MachineODE(layers=layers, app=app)    

t_plot = jnp.linspace(app.t0, app.t1, 200)
B = machine.basis(t_plot)
#plot_bases(t_plot, B, title="Base initiale")

n_colloc = 40
t_colloc = jnp.linspace(app.t0, app.t1, n_colloc)

machine.params, opt_state = train(machine.params, machine, t=t_colloc)

# Visu
u_pred = machine.forward(t_plot)
u_true = app.u_true(t_plot)
B = machine.basis(t_plot)
#plot_bases(t_plot, B, title="Base adaptée")

plot_solutions(t_plot, t_colloc, u_pred, u_true)

print(f"Relative L2 error: {compute_error(app, machine):.3e}")

