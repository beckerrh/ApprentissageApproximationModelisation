import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

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
def train(params, machine, optimizer=None, n_epochs=100, print_every=10, learning_rate=0.1, **kwargs):
    """
    Train parameters using the given optimizer and loss function.
    
    Args:
        params: pytree of parameters to optimize
        loss_fn: function(params) -> scalar loss
        optimizer: optax optimizer instance
        n_epochs: number of training epochs
        print_every: how often to print progress
    
    Returns:
        params: optimized parameters
        opt_state: final optimizer state
    """
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

    return params, opt_state

# -------------------------------------------------------------------------------- #
# MLP-basis
class MLPBasis:
    def __init__(self, layers):
        self.layers = layers
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        params = []
        for l in range(1, len(layers)):
            in_dim, out_dim = layers[l-1], layers[l]
            key, subkey = jax.random.split(key)
            W = jax.random.normal(subkey, (out_dim, in_dim)) * jnp.sqrt(2/in_dim)
            b = jnp.zeros(out_dim)
            params.append((W,b))
        self.params = params
    def __call__(self, t, params=None):
        if params is None:
            params = self.params
        t = jnp.atleast_1d(t).reshape(1, -1)
        for param in params:
            W, b = param
            t = jnp.tanh(W @ t + b[:, None])
        return t

# Machine
class MachineODE:
    def __init__(self, layers, app):
        self.layers = layers
        self.app = app
        self.basis = MLPBasis(layers[:-1])
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        in_dim, out_dim = layers[-2], layers[-1]
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (out_dim, in_dim)) * jnp.sqrt(2/in_dim)*0.1
        b = jnp.zeros(out_dim)
        self.params = (W,b)

        # params = []
        # for l in range(1, len(layers)):
        #     in_dim, out_dim = layers[l-1], layers[l]
        #     key, subkey = jax.random.split(key)
        #     W = jax.random.normal(subkey, (out_dim, in_dim)) * jnp.sqrt(2/in_dim)
        #     b = jnp.zeros(out_dim)
        #     params.append((W,b))
        # self.params = params
    # def basis(self, t, params=None):
    #     if params is None:
    #         params = self.params
    #     t = jnp.atleast_1d(t).reshape(1, -1)
    #     for param in params[:-1]:
    #         W, b = param
    #         t = jnp.tanh(W @ t + b[:, None])
    #     return t
    def forward(self, t, params=None):
        if params is None:
            B = self.basis(t)
            W, b = self.params
            return (W @ B +b).squeeze()
        B = self.basis(t, params['basis'])
        W, b = params['coeff']
        return (W @ B +b).squeeze()
    def residual(self, params, t):
        dudt_scalar = jax.grad(self.forward, argnums=0)
        dudt = jax.vmap(lambda ti: dudt_scalar(ti, params))
        return dudt(t) - self.app.f(self.forward(t, params))
    def loss(self, params, t):
        ode_loss = jnp.mean(self.residual(params, t)**2)
        bc_loss = (self.forward(self.app.t0, params)-self.app.u0) ** 2 
        return ode_loss + bc_loss


# -------------------------------------------------------------------------------- #
class Application:
    def __init__(self):
        self.t0, self.t1 = 0.0, 3.0
        self.u0 = 1.0
        self.lam = 1.0
    def u_true(self, t):
        return jnp.exp(self.lam*t)
    def f(self, u):
        return self.lam * u


app = Application()
layers = [1, 4, 4, 1]
machine = MachineODE(layers=layers, app=app)    

t_plot = jnp.linspace(app.t0, app.t1, 200)
B = machine.basis(t_plot)
#plot_bases(t_plot, B, title="Base initiale")

n_colloc = 20
t_colloc = jnp.linspace(app.t0, app.t1, n_colloc)


params = {'basis': machine.basis.params, 'coeff': machine.params}
lv, g = jax.value_and_grad(lambda p: machine.loss(p, t_colloc))(params)
print(jax.tree.map(lambda x: jnp.linalg.norm(x), g))
params, opt_state = train(params, machine, learning_rate=0.01, n_epochs=400, print_every=100, t=t_colloc)
machine.basis.params = params['basis']
machine.params = params['coeff']


# Visu
u_pred = machine.forward(t_plot)
u_true = app.u_true(t_plot)
B = machine.basis(t_plot)
#plot_bases(t_plot, B, title="Base adaptée")

plot_solutions(t_plot, t_colloc, u_pred, u_true)

