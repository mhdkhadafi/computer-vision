import math
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'muhammadkhadafi'


def getImageGradientandTheta(im):
    sobel_x = np.array([[-1, 0,  1], [-2, 0, 2], [-1, 0,  1]])
    sobel_y = np.array([[1, 2,  1], [0, 0, 0], [-1, -2,  -1]])

    image_gx = signal.convolve2d(im, sobel_x)
    image_gy = signal.convolve2d(im, sobel_y)

    image_gradient = np.sqrt(np.multiply(image_gx, image_gx) + np.multiply(image_gy, image_gy))
    image_theta = np.arctan2(image_gx, image_gy) * 180 / np.pi

    return image_gradient, image_theta


def storeBestCircle(hough, number):
    bestCircles = ([])

    for x in range(number):
        loc = np.where(hough == hough.max())
        circle = [loc[1][0], loc[0][0]]
        bestCircles = np.append(bestCircles, circle)
        hough[loc[0][0], loc[1][0]] = 0

    bestCircles.shape = (number, 2)
    return bestCircles


def hough(im, r=0, bins=1, draw=False):
    if draw:
        plt.clf() # clear the figure

    if r == 0:
        im_diagonal = math.sqrt(im.shape[0]*im.shape[0] + im.shape[1]*im.shape[1])
        hough_bins = np.zeros((math.ceil(im.shape[0] / float(bins)),
                               math.ceil(im.shape[1] / float(bins)),
                               math.ceil(math.ceil(im_diagonal) / float(bins))))
    else:
        hough_bins = np.zeros((math.ceil(im.shape[0] / float(bins)),
                               math.ceil(im.shape[1] / float(bins))))

    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            if im[x, y] == 1:
                if draw:
                    hough_x = ([])
                    hough_y = ([])
                for x0 in range(im.shape[0]):
                    if r == 0:
                        for y0 in range(im.shape[1]):
                            domain = (x-x0)*(x-x0) + (y-y0)*(y-y0)
                            if domain < 0:
                                continue
                            radius= math.sqrt(domain)
                            hough_bins[x0, y0, int(round(radius))] += 1
                    else:
                        if r*r - (x-x0)*(x-x0) < 0:
                            continue
                        y0 = y - math.sqrt(r*r - (x-x0)*(x-x0))
                        hough_bins[x0 / bins, int(round(y0)) / bins] += 1

                    if draw:
                        hough_x = np.append(hough_x, x0)
                        hough_y = np.append(hough_y, y0)
                if draw:
                    p = np.polyfit(hough_x, hough_y, 2)
                    plt.plot(hough_x, np.polyval(p, hough_x), 'r-')
    if draw:
        plt.savefig("hough_space.jpg")
    return hough_bins


def roundAngle(angle_array):
    for i in range(angle_array.shape[0]):
        for j in range(angle_array.shape[1]):
            x = angle_array[i, j]
            if -22.5 < x < 22.5 or x > 157.5 or x < -157.5:
                angle_array[i, j] = 0
            elif 22.5 <= x <= 67.5 or -157.5 <= x <= -112.5:
                angle_array[i, j] = 45
            elif 67.5 < x < 112.5 or -112.5 < x < -67.5:
                angle_array[i, j] = 90
            else:
                angle_array[i, j] = 135
    return angle_array


def supressNonMax(im, thet):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if thet[i, j] == 0:
                if i == 0:
                    if im[i, j] < im[i+1, j]:
                        im[i, j] = 0
                elif i == im.shape[0]-1:
                    if im[i, j] < im[i-1, j]:
                        im[i, j] = 0
                else:
                    if im[i, j] < im[i-1, j] or im[i, j] < im[i+1, j]:
                        im[i, j] = 0
            elif thet[i, j] == 90:
                if j == 0:
                    if im[i, j] < im[i, j+1]:
                        im[i, j] = 0
                elif j == im.shape[1]-1:
                    if im[i, j] < im[i, j-1]:
                        im[i, j] = 0
                else:
                    if im[i, j] < im[i, j+1] or im[i, j] < im[i, j-1]:
                        im[i, j] = 0
            elif thet[i, j] == 45:
                if i == 0:
                    if j != 0:
                        if im[i, j] < im[i+1, j-1]:
                            im[i, j] = 0
                    continue
                if j == 0:
                    if i != 0:
                        if im[i, j] < im[i-1, j+1]:
                            im[i, j] = 0
                    continue
                if i == im.shape[0]-1:
                    if j != im.shape[1]-1:
                        if im[i, j] < im[i-1, j+1]:
                            im[i, j] = 0
                    continue
                if j == im.shape[1]-1:
                    if i != im.shape[0]-1:
                        if im[i, j] < im[i+1, j-1]:
                            im[i, j] = 0
                    continue
                else:
                    if im[i, j] < im[i-1, j+1] or im[i, j] < im[i+1, j-1]:
                        im[i, j] = 0
            elif thet[i, j] == 135:
                if i == 0:
                    if j != im.shape[1]-1:
                        if im[i, j] < im[i+1, j+1]:
                            im[i, j] = 0
                    continue
                if j == 0:
                    if i != im.shape[0]-1:
                        if im[i, j] < im[i+1, j+1]:
                            im[i, j] = 0
                    continue
                if i == im.shape[0]-1:
                    if j != 0:
                        if im[i, j] < im[i-1, j-1]:
                            im[i, j] = 0
                    continue
                if j == im.shape[1]-1:
                    if i != 0:
                        if im[i, j] < im[i-1, j-1]:
                            im[i, j] = 0
                    continue
                else:
                    if im[i, j] < im[i-1, j-1] or im[i, j] < im[i+1, j+1]:
                        im[i, j] = 0

    return im


def thresholding(im, high):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            x = im[i, j]
            if x > high:
                im[i, j] = 1
            else:
                im[i, j] = 0
    return im