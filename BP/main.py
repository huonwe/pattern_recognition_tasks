import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap

import flax.linen as nn
import flax.training.train_state as train_state
import flax

import optax

import pickle

from model import CNN

key = jax.random.PRNGKey(0)

model = CNN()

from torchvision.datasets import MNIST
from data import NumpyLoader, FlattenAndCast, numpy_collate
# 借助于torchvision和NumpyLoader
mnist_dataset_train = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
key, loader_key = jax.random.split(key)
train_loader = NumpyLoader(mnist_dataset_train, loader_key, batch_size=32, shuffle=True,
                           num_workers=0, collate_fn=numpy_collate, drop_last=True)

mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False, transform=FlattenAndCast())
eval_loader = NumpyLoader(mnist_dataset_test, batch_size=128, shuffle=False, num_workers=0,
                          collate_fn=numpy_collate, drop_last=False)


# 学习率调度算法
lr_decay_fn = optax.linear_schedule(
        init_value=1e-3,
        end_value=1e-5,
        transition_steps=200,
)

# 直接上Adam
optimizer = optax.adam(
            learning_rate=lr_decay_fn,
)

batch = jnp.ones((32, 28, 28, 1))  # (N, H, W, C) format
params = model.init(key, batch)
out = model.apply(params, batch)
print(out.shape)

state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def train_step(state, x, y):
    """Computes gradients and loss for a single batch."""
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        one_hot = jax.nn.one_hot(y, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss

    grad_fn = value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

jit_train_step = jit(train_step, donate_argnums=(0,))

@jit
def apply_model(state, x):
    """Computes gradients and loss for a single batch."""
    logits = state.apply_fn(state.params, x)
    return jnp.argmax(logits, -1)

def eval_model(state, loader):
    total_acc = 0.
    total_num = 0.
    for x, y in loader:
        y_pred = apply_model(state, x)
        total_num += len(x)
        total_acc += jnp.sum(y_pred == y)
        # print(x[0,:,:,0].shape)
        plt.imshow(x[0,:,:,0],cmap='gray')
        plt.title(f"pred:{y_pred[0]}  truth:{y[0]}")
        plt.show()
    return total_acc / total_num

# try:
#     params_stored = pickle.load(open('./BP/model.params',"rb"))
#     state = flax.serialization.from_state_dict(target=state, state=params_stored)
#     print("state resumed")
# except Exception():
#     pass


import matplotlib.pyplot as plt
for epoch in range(5):
    for idx, (x, y) in enumerate(train_loader):
        state, loss = jit_train_step(state, x, y)
        if idx % 20 == 0:  # evaluation
            train_acc = eval_model(state, train_loader)
            eval_acc = eval_model(state, eval_loader)
            print("Epoch {} loss {}, tacc {}, vacc {}".format(
              epoch, loss, train_acc, eval_acc))
            
            
            state_dict = flax.serialization.to_state_dict(state)
            pickle.dump(state_dict,open('./BP/model.params','wb'))
            
            