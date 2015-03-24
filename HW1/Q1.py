import math
from heplers import storeBestCircle, hough, roundAngle, supressNonMax, thresholding, getImageGradientandTheta

__author__ = 'muhammadkhadafi'

import numpy as np
import scipy.ndimage as ndimage
from scipy import misc
import matplotlib.pyplot as plt

def detectCircles(im, radius, usegradient):

    image = misc.imread(im, True)
    blurred_image = ndimage.gaussian_filter(image, sigma=1)

    image_gradient, image_theta = getImageGradientandTheta(blurred_image)
    print ("finish gradient " + im)

    # round_theta = np.copy(image_theta)
    round_theta = roundAngle(image_theta)
    print ("finish rounding " + im)

    # image_suppressed = np.copy(image_gradient)
    image_suppressed = supressNonMax(image_gradient, round_theta)
    print ("finish suppress " + im)

    if usegradient:
        threshold = 1.33*np.median(image_suppressed[image_suppressed != 0])
    else:
        threshold = 0

    # image_edges = np.copy(image_suppressed)
    image_edges = thresholding(image_suppressed, threshold)
    print ("finish threshold " + im)

    hough_bins = hough(image_edges, round_theta, radius)
    print ("finish hough " + im)

    # circles_center = np.copy(hough_bins)
    circles_center = storeBestCircle(hough_bins, 50)
    print ("finish circles " + im)

    return circles_center


def drawCircles(im, circles, r, bins=1):
    plt.clf()
    image = misc.imread(im)
    plt.imshow(image)
    for center in circles:
        circle = plt.Circle((center[0]*bins + bins/2, center[1]*bins + bins/2), r, color='b', fill=0, linewidth=3)
        plt.gca().add_artist(circle)
    plt.savefig(im.split(".")[0] + "_circles_bin" + str(bins) + ".jpg")


def drawHoughSpace(im, radius):
    image = misc.imread(im, True)
    blurred_image = ndimage.gaussian_filter(image, sigma=1)
    image_gradient, image_theta = getImageGradientandTheta(blurred_image)
    round_theta = roundAngle(image_theta)
    image_suppressed = supressNonMax(image_gradient, round_theta)
    threshold = 1.33*np.median(image_suppressed[image_suppressed != 0])
    image_edges = thresholding(image_suppressed, threshold)

    hough(image_edges, round_theta, radius, 1, True)
    print ("finish hough " + im)


def drawDifferentQuant(im, radius, quant):
    image = misc.imread(im, True)
    blurred_image = ndimage.gaussian_filter(image, sigma=1)
    image_gradient, image_theta = getImageGradientandTheta(blurred_image)
    round_theta = roundAngle(image_theta)
    image_suppressed = supressNonMax(image_gradient, round_theta)
    threshold = 1.33*np.median(image_suppressed[image_suppressed != 0])
    image_edges = thresholding(image_suppressed, threshold)
    print "got edges"

    for each_quant in quant:
        hough_bins = hough(image_edges, round_theta, radius, each_quant)
        circles_center = storeBestCircle(hough_bins, 50)
        drawCircles(im, circles_center, radius, each_quant)
        print "finish hough quant " + str(each_quant)


def detectAllCircles(im):
    image = misc.imread(im, True)
    blurred_image = ndimage.gaussian_filter(image, sigma=1)
    image_gradient, image_theta = getImageGradientandTheta(blurred_image)
    round_theta = roundAngle(image_theta)
    image_suppressed = supressNonMax(image_gradient, round_theta)
    threshold = 1.33*np.median(image_suppressed[image_suppressed != 0])
    print threshold
    image_edges = thresholding(image_suppressed, threshold)

    hough_bins = hough(image_edges, round_theta)

    bestCircles = ([])

    counter = 0
    while True:
        if counter == 50:
            break
        loc = np.where(hough_bins == hough_bins.max())
        circumference = 2*math.pi*loc[2][0]
        if circumference / hough_bins[loc[0][0], loc[1][0], loc[2][0]] > 4:
            hough_bins[loc[0][0], loc[1][0], loc[2][0]] = 0
            continue
        else:
            circle = [loc[1][0], loc[0][0], loc[2][0]]
            bestCircles = np.append(bestCircles, circle)
            hough_bins[loc[0][0], loc[1][0], loc[2][0]] = 0
            counter += 1
    # for x in range(20):
    #     print hough_bins.max()
    #     loc = np.where(hough_bins == hough_bins.max())
    #     circle = [loc[1][0], loc[0][0], loc[2][0]]
    #     bestCircles = np.append(bestCircles, circle)
    #     hough_bins[loc[0][0], loc[1][0], loc[2][0]] = 0

    bestCircles.shape = (50, 3)

    return bestCircles


def drawAllCircles(im, circles):
    plt.clf()
    image = misc.imread(im)
    plt.imshow(image)
    for center in circles:
        circle1 = plt.Circle((center[0], center[1]), center[2], color='b', fill=0, linewidth=3)
        plt.gca().add_artist(circle1)
    plt.savefig(im.split(".")[0] + "_all_circles.jpg")

if __name__ == "__main__":
    # centers = detectCircles('MoonCraters.jpg', 20, True)
    # drawCircles('MoonCraters.jpg', centers, 20)
    # print "best 20 circle radius 20 list - moon crater"
    # print centers
    # print "------"
    #
    # centers = detectCircles('colorful3.png', 40, True)
    # drawCircles('colorful3.png', centers, 40)
    # print "best 20 circle radius 40 list - colorful 3"
    # print centers
    # print "------"
    #
    # centers = detectCircles('ladybug.jpg', 45, True)
    # drawCircles('ladybug.jpg', centers, 45)
    # print "best 20 circle radius 20 list - ladybug"
    # print centers
    # print "------"
    #
    # centers = detectCircles('colorful2.jpg', 45, True)
    # drawCircles('colorful2.jpg', centers, 45)
    # print "best 20 circle radius 45 list - colorful 2"
    # print centers
    # print "------"

    centers = detectCircles('Planets.jpeg', 300, True)
    drawCircles('Planets.jpeg', centers, 300)
    print "best 20 circle radius 150 list - planets"
    print centers
    print "------"

    # # TODO - too many lines right now, can't see nothing
    # drawHoughSpace('MoonCraters.jpg', 20)
    #
    # drawDifferentQuant('MoonCraters.jpg', 20, [1, 10, 20])

    # # TODO - probably still wrong, need to check with actual pictures
    centers_with_radius = detectAllCircles('MoonCraters.jpg')
    print "best 20 circle all radius list - moon"
    print centers_with_radius
    drawAllCircles('MoonCraters.jpg', centers_with_radius)

    centers_with_radius = detectAllCircles('colorful3.png')
    print "best 20 circle all radius list - moon"
    print centers_with_radius
    drawAllCircles('colorful3.png', centers_with_radius)
    #
    # centers_with_radius = detectAllCircles('Planets.jpeg')
    # print "best 20 circle all radius list - planets"
    # print centers_with_radius
    # drawAllCircles('Planets.jpeg', centers_with_radius)
