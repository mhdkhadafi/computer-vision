
__author__ = 'muhammadkhadafi & daijing'
import os
import sys
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.spatial.distance import euclidean
from PIL import Image
import cv2

from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny, hog

# Check if rectangles overlap
def overlap(ul1, br1, ul2, br2):
    # print ul1, br1, ul2, br2
    if ul1[0] > br2[0] or ul2[0] > br1[0]:
        return False

    if ul1[1] > br2[1] or ul2[1] > br1[1]:
        return False

    return True

# Simple k-Nearest Neighbor used for SIFT (from assignment 2)
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

# Ratio test based on Lowe, to prevent the 2 nearest neighbor to be too similar
def ratio_test(mat, arg):
    good = []
    for i in range(0, arg.shape[1]):
        if mat[i][arg[0][i]] < 0.75*mat[i][arg[1][i]]:
            good.append((i, arg[0][i]))
    return good

# Find the good matches using k-NN
def find_matches(kpt1, kpt2, d1, d2):
    matches, arg_minimum = my_knn(d1, d2, 2)
    good_matches = ratio_test(matches, arg_minimum)
    im1_coord = [kpt1[j[0]].pt for j in good_matches]
    im2_coord = [kpt2[j[1]].pt for j in good_matches]
    return im1_coord, im2_coord


# Get the matching Hough circles from the images
def get_matches(top, im, interval, peak, template):

    # Load picture and detect edges
    image = cv2.imread(im,0)

    # Just to make sure huge pictures don't take up too much time
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        multiplier = 3
    else:
        multiplier = 1
    image = cv2.resize(image, (image.shape[1] / multiplier, image.shape[0] / multiplier), interpolation=cv2.INTER_AREA)
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    # Find the larger side of the image to act as upper limit of circle deection
    smaller_side = image.shape[0]
    longer_side = image.shape[1]
    if smaller_side > image.shape[1]:
        longer_side = image.shape[0]

    # Find circles with radius from 20 until the longest side
    hough_radii = np.arange(20, longer_side, interval)
    # Do the skimage hough_circle function
    hough_res = hough_circle(edges, hough_radii)

    centers = []
    accums = []
    radii = []

    # Only select the top 'peak' matches of the circles in each radius
    for radius, h in zip(hough_radii, hough_res):
        num_peaks = peak
        peaks = peak_local_max(h, num_peaks=num_peaks)
        if len(peaks) == 0:
            continue
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    top_circles = np.argsort(accums)[::-1][:top]

    # Read the original image and add the circles
    plt.clf()
    image = misc.imread(im)
    plt.imshow(image)
    rads = []
    cents = []
    for cir in top_circles:
        # print centers[cir], radii[cir]
        rads.append(radii[cir] * multiplier)
        cents.append(centers[cir] * multiplier)
        circle = plt.Circle((centers[cir][1] * multiplier, centers[cir][0] * multiplier), radii[cir] * multiplier, color='b', fill=0, linewidth=3)
        plt.gca().add_artist(circle)


    plt.title(im.split('.')[0])
    plt.savefig("Results/" + im.split('.')[0] + "_circle.png", bbox_inches='tight')
    match_up(template, im, rads, cents)


# Match all the circle area to the template
def match_up(temp, im, rads, cents):
    img = cv2.imread(im, 0)
    img2 = img.copy()
    template = cv2.imread(temp, 0)

    rectangles = []
    sift_matches = []

    # Go through all the circles to get the rectangles and number of SIFT matches
    for i in range(0, len(rads)):
        center_x = cents[i][1]
        center_y = cents[i][0]

        # Pad the image in case the rectangle is exceeding the image
        padded_image = np.lib.pad(img2, ((rads[i], rads[i]), (rads[i], rads[i])), 'constant', constant_values=0)

        # Crop the image according to the rectangle
        cropped_image = padded_image[center_y:center_y+rads[i]*2, center_x:center_x+rads[i]*2]
        ratio = (rads[i]*2) / float(template.shape[0])
        # Resize the remplate to match the rectangle
        resized_template = cv2.resize(template, (int(template.shape[1] * ratio), int(template.shape[0] * ratio)), interpolation=cv2.INTER_AREA)

        # Initiate SIFT detector
        sift = cv2.SIFT()

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(cropped_image,None)
        kp2, des2 = sift.detectAndCompute(resized_template,None)

        # Find all the SIFT matches coordinates
        img1_coord = []
        if kp1 != None and kp2 != None and des1 != None and des2 != None:
            img1_coord, img2_coord = find_matches(kp1, kp2, des1, des2)

        # Collect the top left and bottom right location for the rectangles
        # As well as the number of SIFT matches in that rectangle
        top_left = (center_x-rads[i], center_y-rads[i])
        bottom_right = (center_x+rads[i], center_y+rads[i])
        rectangles.append((top_left, bottom_right, len(img1_coord)))
        sift_matches.append(len(img1_coord))

    # Find average and standard deviation of the SIFT matches
    sift_matches = np.array(sift_matches)
    sift_avg = np.mean(sift_matches)
    sift_std = np.std(sift_matches)

    # Getting the best rectangles based on mean + std
    best_rectangles = []
    for rect in rectangles:
        if rect[2] > sift_avg + sift_std:
            best_rectangles.append(rect)

    plt.clf()
    img_misc = misc.imread(im)

    # Remove the overlapping rectangles
    no_overlap = []
    for i in range(0, len(best_rectangles)):
        overlap_num = []
        for j in range(0, len(best_rectangles)):
            if overlap(best_rectangles[i][0], best_rectangles[i][1], best_rectangles[j][0], best_rectangles[j][1]):
                overlap_num.append(best_rectangles[j])

        # If no overlap, add it
        if len(overlap_num) == 0:
            no_overlap.append(best_rectangles[i])
        # If there is, add the one with largest SIFT matches
        else:
            overlap_num = sorted(overlap_num,key=itemgetter(2),reverse=True)
            if overlap_num[0] not in no_overlap:
                no_overlap.append(overlap_num[0])

    # Draw the non overlapping rectangle
    for best_rect in no_overlap:
        cv2.rectangle(img_misc,best_rect[0], best_rect[1], 255, 3)

    # Save the images
    plt.imshow(img_misc)
    plt.savefig("Results/" + im.split('.')[0] + "_matches.png", bbox_inches='tight')

# detect polygons (triangles and pentagons) in the image
def detect_shapes(img):
    orig = img.copy()
    imgray = cv2.bilateralFilter(img, 11, 17, 17)
    edged = cv2.Canny(imgray, 30, 200)
    #cv2.imshow('edged', edged)

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    centers=[]
    lengths=[]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05*peri, True)

        if (3<=len(approx)<=5):
            c = np.mean(approx, 0)[0]
            centers.append((int(c[0]),int(c[1])))
            lengths.append(peri)

    return centers, lengths


# Match all the polygon area to the template (for Superman)
def get_supermen(temp, imgfile):
    template = cv2.imread(temp, 0)
    h = template.shape[0]
    w = template.shape[1]
      
    img = cv2.imread(imgfile, 0)

    centers,arclens = detect_shapes(img)
    rectangles = []
    sift_matches = []
    #hog_vals=[]
    for i in range(len(centers)):
        rw = arclens[i]
        center=centers[i]
        ratio = rw/(w+h)

        rh = int(h*ratio/2)
        rw = int(w*ratio/2)

        #Generate 9 rectangles around the center of the polygon
        positions = [(center[1]-rh/2, center[0]-rw/2),
                     (center[1]-rh/2, center[0]),
                      (center[1]-rh/2, center[0]+rw/2),
                      (center[1], center[0]-rw/2),
                      (center[1], center[0]),
                      (center[1], center[0]+rw/2),
                      (center[1]+rh/2, center[0]-rw/2),
                      (center[1]+rh/2, center[0]),
                      (center[1]+rh/2, center[0]+rw/2)]

        resized_template =cv2.resize(template, (2*rw, 2*rh), interpolation=cv2.INTER_AREA)
        resized_template=resized_template.astype('uint8')
        #cv2.imshow('resize', resized_template)
        #cv2.waitKey(0)


        for p in positions:

            if p[0]-rh < 0 or p[1]-rw < 0:
                continue

            # Crop the original image and get the candidate image
            candidate = img[p[0]-rh:p[0]+rh, p[1]-rw:p[1]+rw]
            candidate=candidate.astype('uint8')


            # Initiate SIFT
            sift = cv2.SIFT()
            kp1, des1 = sift.detectAndCompute(candidate, None)
            kp2, des2 = sift.detectAndCompute(resized_template, None)

            img1_coord=[]

            if kp1 != None and kp2 != None and des1 != None and des2 != None:
                img1_coord, img2_coord=find_matches(kp1, kp2, des1, des2)

            top_left = (p[1]-rw, p[0]-rh)
            bottom_right = (p[1]+rw, p[0]+rh)

            rectangles.append((top_left, bottom_right, len(img1_coord)))

            sift_matches.append(len(img1_coord))

    sift_matches=np.array(sift_matches)
    best_value = max(sift_matches)
    best_indices = [i for i, j in enumerate(sift_matches) if j==best_value]

    best_rectangles = []
    for i in best_indices:
        best_rectangles.append(rectangles[i])

    plt.clf()
    img_misc = misc.imread(imgfile)

    # Remove the overlapping rectangles
    no_overlap = []
    for i in range(0, len(best_rectangles)):
        overlap_num = []
        for j in range(0, len(best_rectangles)):
            if overlap(best_rectangles[i][0], best_rectangles[i][1], best_rectangles[j][0], best_rectangles[j][1]):
                overlap_num.append(best_rectangles[j])

        if len(overlap_num) == 0:
            no_overlap.append(best_rectangles[i])
        else:
            overlap_num = sorted(overlap_num,key=itemgetter(2),reverse=True)
            if overlap_num[0] not in no_overlap:
                no_overlap.append(overlap_num[0])

    # Draw the non overlapping rectangle
    for best_rect in no_overlap:
        cv2.rectangle(img_misc,best_rect[0], best_rect[1], 255, 3)

    plt.imshow(img_misc)
    outputname = "Results/"+imgfile.split('.')[0]+ "_matches.png"
    plt.savefig(outputname, bbox_inches='tight')


# Converting gif to png, then to jpg, working with gif is just too messed up
# Credit to fraxel and agconti from stackoverflow
def processGif(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print "Cant load", infile
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save(infile.split('.')[0]+'.png')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

    processPng(infile.split('.')[0]+'.png')

# Converting form png to jpg
# Credit to agconti from stackoverflow
def processPng(infile):

    im = Image.open(infile)
    bg = Image.new("RGB", im.size, (255,255,255))
    bg.paste(im, (0,0), im)
    bg.save(infile.split('.')[0] + '.jpg', quality=95)


if __name__ == '__main__':

    if not os.path.exists("Results"):
        os.makedirs("Results")
        os.makedirs("Results/roadsign")
        os.makedirs("Results/Starbucks")
        os.makedirs("Results/Superman")

    
    # Do the whole thing with road sign
    for i in range(1, 13):
        if i == 11:
            continue
        get_matches(20, 'roadsign/image' + str(i) + '.jpg', 5, 5, 'roadsign/template.jpg')
    get_matches(20, 'roadsign/imag11.jpg', 5, 5, 'roadsign/template.jpg')
    get_matches(20, 'roadsign/lollipop-man.jpg', 5, 5, 'roadsign/template.jpg')

    processGif('Starbucks/template.gif')
    # Do the whole thing with Starbucks
    for i in range(1, 17):
        get_matches(20, 'Starbucks/image' + str(i) + '.jpg', 5, 5, 'Starbucks/template.jpg')


    # Do the whole thing with Superman
    processPng('Superman/template.png')

    for i in range(1, 10):
        if i == 3:
            continue
        get_supermen('Superman/template.jpg', 'Superman/image'+str(i)+'.jpg')
    get_supermen('Superman/template.jpg', 'Superman/image3.jpg')
