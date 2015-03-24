import math
import os

import numpy as np
import scipy.ndimage as ndimage
from scipy import misc
import matplotlib.pyplot as plt

from heplers import storeBestCircle, hough, roundAngle, supressNonMax, thresholding, getImageGradientandTheta, getImageEdgesAndTheta


__author__ = 'muhammadkhadafi'

NUM_CIRCLES = 50


def detectCircles(im, radius, usegradient):
    # Read the image and flatten them into greyscale, then blur them with gaussian filter
    image = misc.imread(im, True)
    blurred_image = ndimage.gaussian_filter(image, sigma=1)

    # Get the image gradient and theta from the blurred image
    # See helpers.py for definition of getImageGradientandTheta()
    image_gradient, image_theta = getImageGradientandTheta(blurred_image)
    print ("finish gradient " + im)

    # Get the rounded value of theta
    # See helpers.py for definition of roundAngle()
    round_theta = roundAngle(image_theta)
    print ("finish rounding " + im)

    # Do a non-maximum suppression on the image
    # See helpers.py for definition of supressNonMax()
    image_suppressed = supressNonMax(image_gradient, round_theta)
    print ("finish suppress " + im)

    if usegradient:
        # Give threshold of 1.33*median(non zero element of suppressed image) if using gradient
        threshold = 1.33*np.median(image_suppressed[image_suppressed != 0])
    else:
        # Use threshold 0 otherwise
        threshold = 0

    # Do a image thresholding on suppressed image, 1 for edge, 0 for not
    # See helpers.py for definition of thresholding()
    image_edges = thresholding(image_suppressed, threshold)
    print ("finish threshold " + im)

    # Do Hough transform on the image edges, returning the bins from the Hough space
    # See helpers.py for definition of hough()
    hough_bins = hough(image_edges, round_theta, radius)
    print ("finish hough " + im)

    # Get the top NUM_CIRCLES circle from the Hough bins
    # See helpers.py for definition of storeBestCircle()
    circles_center = storeBestCircle(hough_bins, NUM_CIRCLES)
    print ("finish circles " + im)

    # Return Nx2 array of x and y coordinate of the center of circles
    return circles_center


# Draw the circles on top of the original image using the Nx2 circle centers
def drawCircles(im, circles, r, useGradient, bins=1):
    plt.clf()
    image = misc.imread(im)
    plt.imshow(image)
    # For all the centers, draw circle of radius r
    # If there bins, draw the center at the center of the bins
    for center in circles:
        circle = plt.Circle((center[0]*bins + bins/2, center[1]*bins + bins/2), r, color='b', fill=0, linewidth=3)
        plt.gca().add_artist(circle)
    plt.title(im + " (bins=" + str(bins) + ", useGradient=" + str(useGradient) + ", radius=" + str(r) + ")")
    plt.savefig("Q1Results/" + im.split(".")[0] + "_" + str(bins) + "_" + str(useGradient) + "_" + str(r) + ".png", bbox_inches='tight')


# Function to draw the Hough space of an image
def drawHoughSpace(im, radius):
    # All the same things that happened in detectCircles with useGradient, combined in 1 function
    # See helpers.py for definition of getImageEdgesAndTheta()
    image_edges, round_theta = getImageEdgesAndTheta(im)

    # Use the same function that puts votes in Hough space, but displaying the Hough space instead
    # See helpers.py for definition of hough()
    hough(image_edges, round_theta, radius, 1, True)
    print ("finish hough " + im)


# Function to draw the different qiantization of Hough space
def drawDifferentQuant(im, radius, quant):
    # All the same things that happened in detectCircles with useGradient, combined in 1 function
    # See helpers.py for definition of getImageEdgesAndTheta()
    image_edges, round_theta = getImageEdgesAndTheta(im)
    print "got edges"

    # For each quantization, find the circles and draw them
    for each_quant in quant:
        # See helpers.py for definition of hough() and storeBestCircle()
        hough_bins = hough(image_edges, round_theta, radius, each_quant)
        circles_center = storeBestCircle(hough_bins, 50)
        drawCircles(im, circles_center, radius, True, each_quant)
        print "finish hough quant " + str(each_quant)


# Function to detect all circles regardless of the radius
# We are only using bins=1 and usegradient=True for this function
def detectAllCircles(im, resize):
    # All the same things that happened in detectCircles with useGradient, combined in 1 function
    # See helpers.py for definition of getImageEdgesAndTheta()
    image_edges, round_theta = getImageEdgesAndTheta(im, resize)

    # Using Hough() withour radius
    # See helpers.py for definition of hough()
    hough_bins = hough(image_edges, round_theta)

    # Find the NUM_CIRCLES best circles
    bestCircles = ([])
    counter = 0
    while True:
        if counter == NUM_CIRCLES:
            break
        loc = np.where(hough_bins == hough_bins.max())
        circumference = 2*math.pi*loc[2][0]
        # While ignoring circles where circumference is more than four times it's votes
        if circumference / hough_bins[loc[0][0], loc[1][0], loc[2][0]] > 4:
            hough_bins[loc[0][0], loc[1][0], loc[2][0]] = 0
            continue
        else:
            circle = [loc[1][0], loc[0][0], loc[2][0]]
            bestCircles = np.append(bestCircles, circle)
            hough_bins[loc[0][0], loc[1][0], loc[2][0]] = 0
            counter += 1

    bestCircles.shape = (NUM_CIRCLES, 3)

    # Return Nx3 array of x, y and r
    return bestCircles


# Function to draw top circles regardless of the radius
# Similar to drawCircles(), however the radius is variable
# We are only using bins=1 and usegradient=True for this function
def drawAllCircles(im, circles):
    plt.clf()
    image = misc.imread(im)
    plt.imshow(image)
    for center in circles:
        circle1 = plt.Circle((center[0], center[1]), center[2], color='b', fill=0, linewidth=3)
        plt.gca().add_artist(circle1)
    plt.title(im + " (radius=all)")
    plt.savefig("Q1Results/" + im.split(".")[0] + "_all_circles.jpg")

if __name__ == "__main__":
    # Create the results directory
    if not os.path.exists("Q1Results"):
        os.makedirs("Q1Results")

    # Draw all 5 images with circles, with and without gradients

    # centers = detectCircles('MoonCraters.jpg', 20, True)
    # drawCircles('MoonCraters.jpg', centers, 20, True)
    # centers = detectCircles('MoonCraters.jpg', 20, False)
    # drawCircles('MoonCraters.jpg', centers, 20, False)

    # centers = detectCircles('colorful3.png', 40, True)
    # drawCircles('colorful3.png', centers, 40, True)
    # centers = detectCircles('colorful3.png', 40, False)
    # drawCircles('colorful3.png', centers, 40, False)
    #
    # centers = detectCircles('ladybug.jpg', 45, True)
    # drawCircles('ladybug.jpg', centers, 45, True)
    # centers = detectCircles('ladybug.jpg', 45, False)
    # drawCircles('ladybug.jpg', centers, 45, False)
    #
    # centers = detectCircles('colorful2.jpg', 45, True)
    # drawCircles('colorful2.jpg', centers, 45, True)
    # centers = detectCircles('colorful2.jpg', 45, False)
    # drawCircles('colorful2.jpg', centers, 45, False)
    #
    # centers = detectCircles('Planets.jpeg', 300, True)
    # drawCircles('Planets.jpeg', centers, 300, True)
    # centers = detectCircles('Planets.jpeg', 300, False)
    # drawCircles('Planets.jpeg', centers, 300, False)

    # Draw the Hough space
    # drawHoughSpace('MoonCraters.jpg', 20)

    # Draw multiple quantization of the 'MoonCraters.jpg'
    # drawDifferentQuant('MoonCraters.jpg', 20, [1, 10, 20, 100])

    # Find circles of and radius from 'MoonCraters.jpg'
    centers_with_radius = detectAllCircles('MoonCraters.jpg', 0.1)
    drawAllCircles('MoonCraters.jpg', centers_with_radius)
