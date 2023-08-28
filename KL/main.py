import os
import cv2
import jax.numpy as jnp
from jax import device_put, jit

import numpy as np

import pickle
import matplotlib.pyplot as plt

import utils

# input: squence_len, n
@jit
def kl_transform(X):
    # print(X.shape)
    face_mean = np.mean(X,axis=1) # 112*92

    P = X.transpose(1,0) - face_mean # 112*92,200
    P = P.transpose(1,0) # 112*92, 200

    P = jnp.array(P)
    # print(P.shape)
    ###
    C1 = jnp.matmul(P,P.T)
    
    w,v = jnp.linalg.eigh(C1)
    # plt.imshow(np.array(v[:,-1].reshape(112,92)), cmap='gray')
    # plt.show()
    ###

    ### another way
    # C = jnp.matmul(P.T,P) # 200, 200
    # u = jnp.matmul(X.reshape(-1,112*92).T,C) # 112*92, 200 * 200,200 -> 112*92, 200
    # for i in range(0,200):
    #     plt.imshow(np.array(u[:, i].reshape(112,92)), cmap='gray')
    #     plt.show()
    ###

    return w, v, face_mean, P

def test(faces, U, mx, P):
    projectFaces = jnp.matmul(U.T , P)
    
    faces = faces.transpose(1,0) - mx
    faces = faces.transpose(1,0)
    projectTestFaces = jnp.matmul(U.T, faces)
    
    correct = 0
    count = 0
    for j in range(0,projectTestFaces.shape[1]):
        distance = []
        for i in range(0,projectFaces.shape[1]):
            dist = jnp.linalg.norm(projectTestFaces[:,j] - projectFaces[:,i])
            distance.append(dist)
        min_dist = min(distance)
        min_dist_idx = distance.index(min_dist)
        
        person_pred = (min_dist_idx // 5) + 1
        person_truth = (j // 5) + 1
        
        print(f"pred: {person_pred} \t truth: {person_truth} \t {person_pred == person_truth}")
        # plt.subplot(3,3,count*3+1)
        # plt.imshow(faces_2[j],cmap='gray')
        # plt.axis('off')
        # plt.title("the face")
        # plt.subplot(3,3,count*3+2)
        # plt.imshow(faces_1[min_dist_idx],cmap='gray')
        # plt.axis('off')
        # plt.title(f"pred person s{person_pred}")
        # plt.subplot(3,3,count*3+3)
        # plt.imshow(faces_1[(person_truth-1)*5],cmap='gray')
        # plt.axis('off')
        # plt.title(f"truth person s{person_truth}")
        # count += 1
        # if count == 3:
        #     count = 0
        #     plt.show()
        #     plt.cla()
        if person_truth == person_pred:
            correct += 1
    print(f"acc: {correct / projectTestFaces.shape[1]}")
    
if __name__ == "__main__":
    faces_1, faces_2 = utils.read_data()
    try:
        f = open("./KL/kl_params.pkl",'rb')
        eigenvalues, eigenvectors, mx, P = pickle.load(f)
        f.close()
        print("cache load success")
    except Exception:
        eigenvalues, eigenvectors, mx, P = kl_transform(faces_1.copy().reshape(200,112*92).transpose(1,0))
        f = open("./KL/kl_params.pkl",'wb')
        pickle.dump([eigenvalues,eigenvectors,mx,P],f)
        f.close()
    
    utils.show_meanfaces(faces_1)
    utils.show_eigenfaces(eigenvectors)

    lamda_sum = jnp.sum(eigenvalues)
    k=0
    thr = 0.8
    sum=0
    for idx in range(1,len(eigenvalues)):
        k+=1
        sum+=eigenvalues[-idx]
        if sum / lamda_sum > thr:
            break
    print("k is ",k) # 33 with thr 0.8
    
    u = eigenvectors[:,-k:]
    test(faces_2.copy().reshape(200,112*92).transpose(1,0), u, mx, P)