import jax.numpy as jnp

import utils

# inputs: [n, feature_size]
def fisher(w1,w2):
    assert w1.shape[1] == w2.shape[1]
    feature_size = w1.shape[1] # 3
    # print(feature_size)
    u1 = jnp.mean(w1,axis=0)
    u2 = jnp.mean(w2,axis=0)
    # print(u1)
    # s1
    s1 = jnp.zeros((feature_size,feature_size))
    # 样本类内离散度矩阵
    for x in w1:
        t_ = x - u1
        s1 += jnp.matmul(t_.reshape(feature_size,1), t_.reshape(1,feature_size))
    # print(s1)
    s2 = jnp.zeros((feature_size,feature_size))
    for x in w2:
        t_ = x - u1
        s2 += jnp.matmul(t_.reshape(feature_size,1), t_.reshape(1,feature_size))
    Sw = s1+s2

    Sb = jnp.matmul((u1-u2).reshape(feature_size,1), (u1-u2).reshape(1,feature_size))
    # print(Sb)
    w_star = jnp.matmul(jnp.linalg.inv(Sw), u1-u2)
    # print(w_star)

    y1 = jnp.matmul(w1,w_star.reshape(feature_size,1))
    y2 = jnp.matmul(w2,w_star.reshape(feature_size,1))
    
    y0 = (jnp.mean(y1)+jnp.mean(y2))/2

    utils.draw3d1(y1,y2)

    return w_star, y0
    
    
    
if __name__ == "__main__":
    w1 = jnp.array([[-0.4, 0.58, 0.089],
                    [-0.31, 0.27, -0.04],
                    [-0.38, 0.055, -0.035],
                    [-0.15, 0.53, 0.011],
                    [-0.35, 0.47, 0.034],
                    [0.17, 0.69, 0.1],
                    [-0.011, 0.55, -0.18],
                    [-0.27, 0.61, 0.12],
                    [-0.065, 0.49, 0.0012],
                    [-0.12, 0.054, -0.063],
                    ])
    w2 = jnp.array([[0.83, 1.6, -0.014],
                    [1.1, 1.6, 0.48],
                    [-0.44, -0.41, 0.32],
                    [0.047, -0.45, 1.4],
                    [0.28, 0.35, 3.1],
                    [-0.39, -0.48, 0.11],
                    [0.34, -0.079, 0.14],
                    [-0.3, -0.22, 2.2],
                    [1.1, 1.2, -0.46],
                    [0.18, -0.11, -0.49],
                    ])

    utils.draw3d(w1,w2)

    w,y0 = fisher(w1,w2)
    
    x_new = jnp.array([-0.7,0.58,0.089])
    y_new = jnp.matmul(x_new, w.reshape(3,1)) - y0
    print("样本属于w1" if y_new > 0 else "样本属于w2")