import math
from scipy import signal, misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'muhammadkhadafi'


# Function to get image gradient and theta
def getImageGradientandTheta(im):
    # We are usins Sobel mast to convolve the image
    sobel_x = np.array([[-1, 0,  1], [-2, 0, 2], [-1, 0,  1]])
    sobel_y = np.array([[1, 2,  1], [0, 0, 0], [-1, -2,  -1]])

    # Convolving the image
    image_gx = signal.convolve2d(im, sobel_x)
    image_gy = signal.convolve2d(im, sobel_y)

    # Getting the imate gradient; g = sqrt(gx^2 + gy^2)
    image_gradient = np.sqrt(np.multiply(image_gx, image_gx) + np.multiply(image_gy, image_gy))
    # Getting the theta
    image_theta = np.arctan2(image_gx, image_gy) * 180 / np.pi

    return image_gradient, image_theta


# Function to store the top 'number' valued bins as circle's center
def storeBestCircle(hough_bins, number):
    bestCircles = ([])

    for x in range(number):
        # Get the max, find the location, remove the max, next
        loc = np.where(hough_bins == hough_bins.max())
        circle = [loc[1][0], loc[0][0]]
        bestCircles = np.append(bestCircles, circle)
        hough_bins[loc[0][0], loc[1][0]] = 0

    bestCircles.shape = (number, 2)
    return bestCircles


# Function to do Hough transform on the image edges, returning the bins from the Hough space
# It also has other functions that can be activated, such as drawing Hough space when draw=True
# Also when r=0, it will return 3D Hough bins with [x0, y0, r]
def hough(im, thet, r=0, bins=1, draw=False):
    # Prepare to draw
    if draw:
        plt.clf()

    # Measure the domains for constant r and variable r
    # Domain for x0 and y0 is width and height, while r is the diagonal
    if r == 0:
        im_diagonal = math.sqrt(im.shape[0]*im.shape[0] + im.shape[1]*im.shape[1])
        hough_bins = np.zeros((math.ceil(im.shape[0] / float(bins)),
                               math.ceil(im.shape[1] / float(bins)),
                               math.ceil(math.ceil(im_diagonal) / float(bins))))
    else:
        hough_bins = np.zeros((math.ceil(im.shape[0] / float(bins)),
                               math.ceil(im.shape[1] / float(bins))))

    # Loop through image's x and y
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            xy_angle = thet[x, y]
            if im[x, y] == 1:
                if draw:
                    hough_x = ([])
                    hough_y = ([])
                for x0 in range(im.shape[0]):
                    # If r is variable, continue looping y0
                    if r == 0:
                        for y0 in range(im.shape[1]):
                            # Take care of sqrt negative
                            domain = (x-x0)*(x-x0) + (y-y0)*(y-y0)
                            if domain < 0:
                                continue
                            # r^2 = (x-x0)^2 + (y-y0)^2
                            radius = math.sqrt(domain)
                            # check if the angle of (x, y) is perpendicular to line (x,y) to (x0, y0)
                            # by checking if theta is equal
                            angle = roundSingleAngle(math.atan2((y0-y), (x0-x)) * 180 / math.pi)
                            if angle == xy_angle:
                                hough_bins[x0 / bins, y0 / bins, int(round(radius)) / bins] += 1
                    # If r is constant, gey y0
                    else:
                        # Rheck domain
                        if r*r - (x-x0)*(x-x0) < 0:
                            continue
                        # y0 = y - sqrt(r^2 - (x-x0)^2)
                        y0 = y - math.sqrt(r*r - (x-x0)*(x-x0))
                        y0 = int(round(y0))
                        # Check angle here too
                        angle = roundSingleAngle(math.atan2((y0-y), (x0-x)) * 180 / math.pi)
                        if angle == xy_angle:
                            hough_bins[x0 / bins, y0 / bins] += 1

                    # get all x0 and y0 to create a line
                    if draw:
                        hough_x = np.append(hough_x, x0)
                        hough_y = np.append(hough_y, y0)
                # Fit the points and draw them
                if draw:
                    p = np.polyfit(hough_x, hough_y, 1)
                    plt.plot(np.polyval(p, hough_x), hough_x,'y-', alpha=0.5)
                    # plt.gca().invert_yaxis()
    if draw:
        plt.title("MoonCraters.jpg Hough Space")
        plt.gca().invert_yaxis()
        plt.savefig("Q1Results/hough_space.png", bbox_inches='tight')

    # Return the hough space bins
    return hough_bins


# Function to round single angle to 0, 45, 90, 135 specially for checking perpendicular lihe from hough()
def roundSingleAngle(single_angle):
    if -22.5 < single_angle < 22.5 or single_angle > 157.5 or single_angle < -157.5:
        return 0
    elif 22.5 <= single_angle <= 67.5 or -157.5 <= single_angle <= -112.5:
        return 135
    elif 67.5 < single_angle < 112.5 or -112.5 < single_angle < -67.5:
        return 90
    else:
        return 45


# Function to round array of angles to 0, 45, 90, 135, for image theta
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


# Functino to suppress non-maximum
# Basically check the direction of theta, suppress if it's not maximum from the direction perpendicular to theta
# Plenty of ifs to error check the array edges
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


# Function to return thresholded image
# 1 if above threshold, 0 if below
def thresholding(im, high):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            x = im[i, j]
            if x > high:
                im[i, j] = 1
            else:
                im[i, j] = 0
    return im


# Get the image edges
# Basically all the same steps from detectCircles until thresholding an image
# This is used a lot on other questions
def getImageEdgesAndTheta(im, resize=1.):
    image = misc.imread(im, True)
    # Resizing the image is necessary
    image = misc.imresize(image, resize)
    blurred_image = ndimage.gaussian_filter(image, sigma=1)
    image_gradient, image_theta = getImageGradientandTheta(blurred_image)
    round_theta = roundAngle(image_theta)
    image_suppressed = supressNonMax(image_gradient, round_theta)
    threshold = 1.33*np.median(image_suppressed[image_suppressed != 0])
    image_edges = thresholding(image_suppressed, threshold)

    return [image_edges, round_theta]