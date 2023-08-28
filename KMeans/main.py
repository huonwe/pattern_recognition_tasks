import jax.numpy as jnp
from jax import device_put, jit, random

import matplotlib.pyplot as plt

def dist(x1,x2):
    return jnp.sum(jnp.power(x1-x2,2))

w = jnp.array([
    [1.45,-0.38],
    [1.67,0.13],
    [0.74,0.40],
    [1.09,-0.11],
    [1.38,0.24],
    [4.99,6.79],
    [5.14,6.28],
    [5.63,6.32],
    [4.68,5.57],
    [4.68,6.06],
])

w1_center = w[0]
w2_center = w[1]

w1_old = []
w2_old = []

while True:
    w1_new = []
    w2_new = []
    for x in w:
        d1 = dist(x,w1_center)
        d2 = dist(x,w2_center)
        if d1 < d2:
            w1_new.append(x)
        else:
            w2_new.append(x)
    
    w1_center = jnp.mean(jnp.array(w1_new),axis=0)
    w2_center = jnp.mean(jnp.array(w2_new),axis=0)
    print(w1_center,w2_center)
    if (len(w1_old) == len(w1_new)) and (len(w2_old) == len(w2_new)):
        break
    w1_old = w1_new
    w2_old = w2_new
    
for x in w:
    plt.plot(x[0],x[1],color='g',marker='<')
plt.plot(w1_center[0],w1_center[1],color='r',marker='*')
plt.plot(w2_center[0],w2_center[1],color='r',marker='*')
plt.show()