__author__ = 'muhammadkhadafi'

import numpy as np
import cv2
import os
from scipy.spatial.distance import euclidean
from scipy import misc
from matplotlib import pyplot as plt
from numpy.linalg import lstsq, svd
import random
from math import sqrt
from skimage import transform
from skimage.transform import ProjectiveTransform

def my_knn(d1, d2, k):
    mat = np.zeros(len(d1)*len(d2)).reshape(len(d1), len(d2))
    arg = []
    for i in range(0, len(d1)):
        for j in range(0, len(d2)):
            mat[i][j] = euclidean(d1[i], d2[j])
    mat_copy = np.copy(mat)
    for i in range(0, k):
        mat_arg = np.argmin(mat_copy, axis=1)
        for j in range(0, len(mat_arg)):
            mat_copy[j][mat_arg[j]] = mat_copy.max()+1
        arg.append(mat_arg)
    arg = np.array(arg)
    return mat, arg

def ratio_test(mat, arg):
    good = []
    for i in range(0, arg.shape[1]):
        if mat[i][arg[0][i]] < 0.75*mat[i][arg[1][i]]:
            good.append((i, arg[0][i]))
    return good


def find_matches(kpt1, kpt2, d1, d2):
    matches, arg_minimum = my_knn(d1, d2, 2)
    good_matches = ratio_test(matches, arg_minimum)
    im1_coord = [kpt1[j[0]].pt for j in good_matches]
    im2_coord = [kpt2[j[1]].pt for j in good_matches]
    return im1_coord, im2_coord


def show_matches(im1, im2, im1_coord, im2_coord, name="blah"):
    plt.clf()
    plt.axis("off")

    image = np.lib.pad(im1, ((0, im2.shape[0]-im1.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
    # image1.shape
    im = np.append(image, im2, axis=1)
    im.shape
    plt.imshow(im)

    for i in range(0, len(im1_coord)):
        circle1 = plt.Circle((im1_coord[i][0], im1_coord[i][1]), 2, color='y')
        plt.gca().add_artist(circle1)
        circle2 = plt.Circle((im2_coord[i][0]+im1.shape[1], im2_coord[i][1]), 2, color='y')
        plt.gca().add_artist(circle2)
        line = plt.Line2D(xdata=[im1_coord[i][0], im2_coord[i][0]+im1.shape[1]], ydata = [im1_coord[i][1], im2_coord[i][1]], color='y')
        plt.gca().add_artist(line)

    plt.title(name)
    plt.savefig("Q1Results/" + name + ".png",  bbox_inches='tight')


def get_affine_transformation(im1_coord, im2_coord, seed=True):
    a = []
    b = []
    if seed:
        sample = random.sample(range(0, len(im1_coord)), 3)
        for i in range(0, len(sample)):
            a.append([[im1_coord[sample[i]][0], im1_coord[sample[i]][1], 1, 0, 0, 0],
                     [0, 0, 0, im1_coord[sample[i]][0], im1_coord[sample[i]][1], 1]])
            b.append([[im2_coord[sample[i]][0]], [im2_coord[sample[i]][1]]])

        a = np.array(a).reshape(len(sample)*2, 6)
        b = np.array(b).reshape(len(sample)*2, 1)
    else:
        for i in range(0, len(im1_coord)):
            a.append([[im1_coord[i][0], im1_coord[i][1], 1, 0, 0, 0],
                     [0, 0, 0, im1_coord[i][0], im1_coord[i][1], 1]])
            b.append([[im2_coord[i][0]], [im2_coord[i][1]]])
        a = np.array(a).reshape(len(im1_coord)*2, 6)
        b = np.array(b).reshape(len(im1_coord)*2, 1)
    x = lstsq(a, b)

    if x[0][0][0] < 0.001 and x[0][1][0] < 0.001 and x[0][3][0] < 0.001 and x[0][4][0] < 0.001:
        x = [[0,0,0],[0,0,0],[0,0,1]]
    else:
        x = [[x[0][0][0], x[0][1][0], x[0][2][0]],
             [x[0][3][0], x[0][4][0], x[0][5][0]],
             [0, 0, 1]]
    x = np.array(x)
    return x


def is_nearby(p1, p2, threshold):
    dist = sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    if dist <= threshold:
        return True
    else:
        return False

def test_transformation(trans, im1_coord, im2_coord):
    count_inliers = 0
    count_outliers = 0
    inliers = []
    for i in range(0, len(im1_coord)):
        a = np.array([[im1_coord[i][0]], [im1_coord[i][1]], [1]])
        b = np.dot(trans, a)
        near = is_nearby((im2_coord[i][0], im2_coord[i][1]), (b[0][0], b[1][0]), 20)
        if near:
            count_inliers += 1
            inliers.append(i)
        else:
            count_outliers += 1
    return count_inliers, count_outliers, inliers

def my_ransac(tries, im1_coord, im2_coord, trans_type):
    best_inliers = 0
    inliers_points = []
    transformation_matrix = []
    while tries > 0:
        if (trans_type == 'affine'):
            trans = get_affine_transformation(im1_coord, im2_coord)
        else:
            trans = get_projective_transformation(im1_coord, im2_coord)
            trans = trans / trans[-1][-1]
        inliers, outliers, points = test_transformation(trans, im1_coord, im2_coord)
        if best_inliers < inliers:
            best_inliers = inliers
            inliers_points = points
            transformation_matrix = trans
        tries -= 1
    return inliers_points, transformation_matrix

def affine_matches(im1_coord, im2_coord):
    in_points_proj, projection_matrix = my_ransac(10000, im1_coord, im2_coord, 'affine')
    proj_im1_coord = np.array(im1_coord)
    proj_im1_coord = list(proj_im1_coord[in_points_proj])
    proj_im2_coord = np.array(im2_coord)
    proj_im2_coord = list(proj_im2_coord[in_points_proj])
    return proj_im1_coord, proj_im2_coord, projection_matrix


def align_images(im1, im2, trans, name='blah'):

    trans_im1 = np.lib.pad(im1, ((0, im2.shape[0]-im1.shape[0]), (0, im2.shape[1]-im1.shape[1]), (0, 0)), 'constant', constant_values=0)
    trans_im1 = transform.warp(trans_im1, ProjectiveTransform(matrix=trans).inverse)
    trans_im1[:,:,0] = 0
    trans_im1[:,:,1] = 0
    trans_im1_rescaled = []
    for x in np.nditer(trans_im1):
        trans_im1_rescaled.append(x * 255)
    trans_im1_rescaled = np.array(trans_im1_rescaled).reshape(trans_im1.shape).astype('uint8')
    rg_image2 = np.copy(im2)
    rg_image2[:,:,2] = 0
    total_image = trans_im1_rescaled + rg_image2
    plt.clf()
    plt.axis("off")
    plt.imshow(total_image)
    plt.title(name)
    plt.savefig("Q1Results/" + name + ".png",  bbox_inches='tight')

    return total_image

def projective_matches(im1_coord, im2_coord):
    in_points_proj, projection_matrix = my_ransac(10000, im1_coord, im2_coord, 'projective')
    proj_im1_coord = np.array(im1_coord)
    proj_im1_coord = list(proj_im1_coord[in_points_proj])
    proj_im2_coord = np.array(im2_coord)
    proj_im2_coord = list(proj_im2_coord[in_points_proj])
    return proj_im1_coord, proj_im2_coord, projection_matrix

def get_projective_transformation(im1_coord, im2_coord, seed=True):
    a = []
    if seed:
        sample = random.sample(range(0, len(im1_coord)), 6)
        for i in range(0, len(sample)):
            a.append([[im1_coord[sample[i]][0], im1_coord[sample[i]][1], 1, 0, 0, 0, im1_coord[sample[i]][0]*im2_coord[sample[i]][0]*(-1), im1_coord[sample[i]][1]*im2_coord[sample[i]][0]*(-1), im2_coord[sample[i]][0]*(-1)],
                      [0, 0, 0, im1_coord[sample[i]][0], im1_coord[sample[i]][1], 1, im1_coord[sample[i]][0]*im2_coord[sample[i]][1]*(-1), im1_coord[sample[i]][1]*im2_coord[sample[i]][1]*(-1), im2_coord[sample[i]][1]*(-1)]])
        a = np.array(a).reshape(len(sample)*2, 9)
    else:
        for i in range(0, len(im1_coord)):
            a.append([[im1_coord[i][0], im1_coord[i][1], 1, 0, 0, 0, im1_coord[i][0]*im2_coord[i][0]*(-1), im1_coord[i][1]*im2_coord[i][0]*(-1), im2_coord[i][0]*(-1)],
                      [0, 0, 0, im1_coord[i][0], im1_coord[i][1], 1, im1_coord[i][0]*im2_coord[i][1]*(-1), im1_coord[i][1]*im2_coord[i][1]*(-1), im2_coord[i][1]*(-1)]])
        a = np.array(a).reshape(len(im1_coord)*2, 9)

    x = svd(a)
    x = x[2][x[1].shape[0]-1].reshape(3,3)
    # if x[0] < 0.001 and x[1] < 0.001 and x[3] < 0.001 and x[4] < 0.001:
    #     x = [[0,0,0],[0,0,0],[0,0,1]]

    return x


def image_distance(merged_image, im2):
    return euclidean(merged_image.flatten(), im2.flatten())

def q1(img1_loc, img2_loc, img1_name, img2_name):
    print img1_name + " - " + img2_name
    # Initiate SIFT detector
    sift = cv2.SIFT()

    img1 = cv2.imread(img1_loc,0)
    img2 = cv2.imread(img2_loc,0)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    image1 = misc.imread(img1_loc)
    image2 = misc.imread(img2_loc)
    img1_coord, img2_coord = find_matches(kp1, kp2, des1, des2)

    show_matches(image1, image2, img1_coord, img2_coord, img1_name + " - " + img2_name + " all keypoint")
    img1_affine_coord, img2_affine_coord, affine_trans = affine_matches(img1_coord, img2_coord)
    average_affine_trans = get_affine_transformation(img1_affine_coord, img2_affine_coord, False)
    show_matches(image1, image2, img1_affine_coord, img2_affine_coord, img1_name + " - " + img2_name + " affine keypoint")
    affine_aligned_image = align_images(image1, image2, average_affine_trans, img1_name + " - " + img2_name + " affine aligned")
    affine_success_rate = image_distance(affine_aligned_image, image2)
    print "affine failure rate: " + str(affine_success_rate)

    img1_proj_coord, img2_proj_coord, proj_trans = projective_matches(img1_coord, img2_coord)
    show_matches(image1, image2, img1_proj_coord, img2_proj_coord, img1_name + " - " + img2_name + " projective keypoint")
    average_proj_trans = get_projective_transformation(img1_proj_coord, img2_proj_coord, False)
    proj_aligned_image = align_images(image1, image2, average_proj_trans, img1_name + " - " + img2_name + " projective aligned")
    proj_success_rate = image_distance(proj_aligned_image, image2)
    print "projective failure rate: " + str(proj_success_rate)
    print ''

if __name__ == "__main__":
    # Create the results directory
    if not os.path.exists("Q1Results"):
        os.makedirs("Q1Results")

    q1('StopSign1.jpg', 'StopSign2.jpg', "Stop Sign 1", "Stop Sign 2")
    q1('StopSign1.jpg', 'StopSign3.jpg', "Stop Sign 1", "Stop Sign 3")
    q1('StopSign1.jpg', 'StopSign4.jpg', "Stop Sign 1", "Stop Sign 4")




