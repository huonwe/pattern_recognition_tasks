import jax.numpy as jnp
from jax import device_put, jit, random

import numpy as np
import matplotlib.pyplot as plt 

y1 = jnp.array([[0.1, 1.1, 1],
                [6.8, 7.1, 1],
                [-3.5, -4.1, 1],
                [2.0, 2.7, 1],
                [4.1, 2.8, 1],
                [3.1, 5.0, 1],
                [-0.8, -1.3, 1],
                [0.9, 1.2, 1],
                [5.0, 6.4, 1],
                [3.9, 4.0, 1],
                ])

y2 = jnp.array([[7.1, 4.2, 1],
                [-1.4, -4.3, 1],
                [4.5, 0, 1],
                [6.3, 1.6, 1],
                [4.2, 1.9, 1],
                [1.4, -3.2, 1],
                [2.4, -4.0, 1],
                [2.5, -6.1, 1],
                [8.4, 3.7, 1],
                [4.1, -2.2, 1],
                ])

key = random.PRNGKey(0)
w0 = random.normal(key, (3,))

epoch = 500
lr = 1
while epoch > 0:
    print("epoch: ",epoch)
    J = 0
    for a,b in zip(y1,y2):
        b = -b
        g = jnp.dot(w0, a)
        if g < 0:
            w0 += a * lr
            J += -g
        g = jnp.dot(w0, b)
        if g < 0:
            w0 += b * lr
            J += -g
        # print(f"{w0} * {a} = {g}")
    print(J)
    if J < 0.001:
        break
    epoch -= 1
print("weight: ",w0)
for a,b in zip(y1,y2):
    plt.plot(a[0],a[1],color='b',marker='<')
    plt.plot(b[0],b[1],color='g',marker='>')

x1 = jnp.arange(-6,8,0.01)
x2 = (-w0[2] - x1*w0[0]) / w0[1]
plt.plot(x1,x2,color='r')
plt.show()