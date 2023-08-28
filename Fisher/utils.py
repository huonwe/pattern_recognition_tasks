import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw3d(w1,w2):
    ax = plt.figure().add_subplot(projection='3d')
    for x in w1:
        ax.scatter(x[0],x[1],x[2],c='r',marker='^')
    for x in w2:
        ax.scatter(x[0],x[1],x[2],c='g',marker='*')
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    plt.show()
    
def draw3d1(y1,y2):
    ax = plt.figure().add_subplot(projection='3d')
    for y in y1:
        ax.scatter(y,0,0,c='r',marker='^')
    for y in y2:
        ax.scatter(y,0,0,c='g',marker='*')
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    plt.show()