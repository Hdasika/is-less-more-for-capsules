import matplotlib.pyplot as plt

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse


parser = argparse.ArgumentParser(description='MarginLoss Visualization')
parser.add_argument('--lambdaa', metavar='lambda', type=float, required=True, default=0.5, help='enter the lambda value')
parser.add_argument('--mp', metavar='m+', type=float, required=True, default=0.9,help='enter m+')
parser.add_argument('--mn', metavar='m-', type=float, required=True,default=0.1, help='enter m-')

args = parser.parse_args()

l1 = [0,0]
l2 = [0,0]
x = np.linspace(0, 1, 15000)
y = np.linspace(0, 1, 15000)
X, Y = np.meshgrid(x, y)

classes = 2
norm = np.sqrt(X ** 2 + Y ** 2)
for i in range(0, classes):
    if i == 0:
        l1[0] = np.maximum(0.0,args.mp-norm)**2
        #if i == 1:    
        l1[1] = args.lambdaa*np.maximum(0.0,norm-args.mn)**2
    else:
        l2[0] = np.maximum(0.0,args.mp-norm)**2
        #if i == 1:    
        l2[1] = args.lambdaa*np.maximum(0.0,norm-args.mn)**2

L_1 = l1[0] + l2[1]
L_2 = l1[1] + l2[0]

print(L_1.shape, L_2.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, L_1+L_2)

ax.set_xlabel('Class 1')
ax.set_ylabel('Class 2')
ax.set_zlabel('Margin Loss')
plt.show()