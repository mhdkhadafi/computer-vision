__author__ = 'daijing'

import numpy as np

WORLD_INFILE = 'HW2_world.txt'
IMAGE_INFILE = 'HW2_image.txt'

if __name__ == '__main__':
    # read the coordinates of world points
    world = []
    f = open(WORLD_INFILE, 'r')
    world_x = f.readline().split()
    world_y = f.readline().split()
    world_z = f.readline().split()
    for i in range(len(world_x)):
        p = [float(world_x[i]), float(world_y[i]), float(world_z[i]), 1.0]
        world.append(p)
    del world_x, world_y
    f.close()

    # read the coordinates of image points
    image = []
    f2=open(IMAGE_INFILE, 'r')
    image_x = f2.readline().split()
    image_y = f2.readline().split()
    for j in range(len(image_x)):
        q = [float(image_x[j]), float(image_y[j]), 1.0]
        image.append(q)
    del image_x, image_y
    f2.close()

    # Question 1.

    # construct the matrix A
    A = []
    for i in range(len(world)):
        row1 = [0.0, 0.0, 0.0, 0.0] + [-image[i][2]*x for x in world[i]] + [image[i][1]*x for x in world[i]]
        A.append(row1)

        row2 = [image[i][2]*x for x in world[i]] + [0.0, 0.0, 0.0, 0.0] + [-image[i][0]*x for x in world[i]]
        A.append(row2)
    print A
    U, s, V = np.linalg.svd(A, full_matrices = True) #values in s are sorted in descending order

    # the right eigenvector corresponding to the smallest singular value of A
    vind = np.argmin(s)
    p = V[vind, :]
    camera = p.reshape(3,4)
    print "Projection matrix: ", camera
    for i in range(10):
        print i+1, "Original image:", image[i][:2]
        x=np.array(world[i])
        x_img = np.dot(camera, x)
        print "Reprojection:", [x_img[0]/x_img[2], x_img[1]/x_img[2]]
        print "=="

    # Question 2.
    # compute the projection center of the camera C
    Up, sp, Vp = np.linalg.svd(camera, full_matrices=True)
    cind = np.argmin(sp)
    c = Vp[cind, :]
    cw = c/c[3]
    cw = cw[0:3] #world coordinates of the camera centre C
    print "Projection center:", cw

