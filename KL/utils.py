import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_data():
    faces_1 = np.zeros((200,112,92),dtype=np.float32)
    faces_2 = np.zeros((200,112,92),dtype=np.float32)
    root = './KL/orl_faces'
    for dir in os.listdir(root):
        path_people = os.path.join(root,dir)
        for file in os.listdir(path_people):
            path_file = os.path.join(path_people, file)
            face = cv2.imread(path_file,0)
            if int(file.split('.')[0]) in [1,2,3,4,5]:
                face = cv2.imread(path_file,0)
                # face = cv2.equalizeHist(face)
                idx = (int(dir[1:]) - 1)*5 + int(file.split('.')[0]) - 1
                faces_1[idx] = face
            else:
                idx = (int(dir[1:]) - 1)*5 + int(file.split('.')[0]) - 6
                faces_2[idx] = face
                # if dir == "s1" and file =="6.pgm":
                #     print(idx,dir,file)
                #     plt.imshow(faces_2[idx],cmap='gray')
                #     plt.show()
    # os._exit(0)
                
    # for i in range(0,10):
    #     # plt.subplot(1,2,1)
    #     # plt.imshow(faces_1[i],cmap='gray')
    #     # plt.subplot(1,2,2)
    #     plt.imshow(faces_2[i],cmap='gray')
    #     plt.show()
    return faces_1, faces_2

def show_eigenfaces(U):
    plt.cla()
    for i in range(0,5):
        plt.subplot(1,5,i+1)
        plt.imshow(np.array(U[:,-i-1].reshape(112,92)), cmap='gray')
    plt.show()

def show_meanfaces(faces):
    mean = np.mean(faces,axis=0)
    plt.imshow(mean,cmap='gray')
    plt.show()

# def idx2path(idx):
    