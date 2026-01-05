On s'intéresse à l'EDO sur $[0,3]$
$$
\frac{d^2u}{dt^2} +\pi^2 \sin(\pi t) = 0,\quad u(0)=0=u(3).
$$
On se sert uniquement de **JAX**


```python
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

# Commençons par la fonction de visu
def plot_solutions(t_plot, u_pred, u_true, t_colloc):
    plt.plot(t_plot, u_pred, label='Approximation')
    plt.plot(t_plot, u_true, '--', label='Solution')
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='t_colloc')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(r"$u'' = -\pi^2 \sin(\pi t)$")
    plt.grid(True)
    plt.show()

# Points de collocation
t0, t1, n_colloc = 0, 3, 10
t_colloc = jnp.linspace(t0, t1, n_colloc)
# Machine
layers = [1, 8, 8, 1]
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
# Initialisation des paramètres
params = []
for l in range(1,len(layers)):
    in_dim, out_dim =  layers[l-1], layers[l]
    key, subkey = jax.random.split(key)
    W = jax.random.normal(subkey, (out_dim, in_dim)) * jnp.sqrt(2 / in_dim)
    b = jnp.zeros(out_dim)
    params.append((W,b))
def forward(params, t):
    t = jnp.array([t])
    for W, b in params[:-1]:
        t = jnp.tanh(W @ t + b)
    W, b = params[-1]
    return (W @ t +b)[0]
# Calcul de u''(x) par JAX autodiff
def dudt(params, t):
    return jax.grad(forward, argnums=1)(params, t)
def d2udt2(params, t):
    return jax.grad(dudt, argnums=1)(params, t)
def residual( params, t):
    return d2udt2(params, t) + (jnp.pi ** 2) * jnp.sin(jnp.pi * t)
# Total loss = physics + boundary conditions
def loss(params):
    # Physics loss
    res = jax.vmap(lambda t: residual(params, t))(t_colloc)
    physics_loss = jnp.mean(res ** 2)
    # Boundary conditions
    bc_loss = forward(params, t_colloc[0]) ** 2 + forward(params, t_colloc[-1]) ** 2
    return physics_loss + bc_loss

# Une itération de gradient
@jax.jit
def update(params, lr):
    grads = jax.grad(loss)(params)
    return  [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
# Boucle d'entraînement
for epoch in range(6000):
    params = update(params, lr=0.001)
    if epoch % 300 == 0:
        loss_val = loss(params)
        print(f"{epoch:6d} {loss_val:12.3e}")


# Visu
t_plot = jnp.linspace(t0, t1, 200)
u_pred = jax.vmap(lambda t: forward(params,t))(t_plot)
u_true = jnp.sin(jnp.pi * t_plot)

plot_solutions(t_plot, u_pred, u_true, t_colloc)
```

         0    4.269e+01
       300    2.147e+01
       600    1.905e+01
       900    1.213e+01
      1200    5.309e+00
      1500    4.723e+00
      1800    4.545e+00
      2100    4.299e+00
      2400    2.145e+00
      2700    5.100e-01
      3000    1.036e-01
      3300    1.910e-02
      3600    3.917e-03
      3900    9.824e-04
      4200    2.887e-04
      4500    9.109e-05
      4800    2.942e-05
      5100    9.542e-06
      5400    3.109e-06
      5700    1.009e-06



    
![png](ode2o_files/ode2o_1_1.png)
    


C'est un exemple basique avec des outls basiques.
On peut utiliser des outils plus performant comme les librairies :
* **FLAX** définit des réseaux de neuronnes
* **OPTAX** donne accès à un grand nombre d'algorithmes d'optimisation


