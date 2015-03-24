__author__ = 'daijing'

from matplotlib.pylab import imread, imshow, figure, show, savefig,title
from numpy import reshape, array, uint8, delete
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import colorsys
import math
import os

#Part 1:
def quantizeRGB(img, clusters, name):
    #reshape the pixels matrix
    pixel = reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    #perform clustering
    centroids,_ = kmeans(pixel, clusters)

    #quantization
    qnt,_ = vq(pixel, centroids)

    #reshape the result of the quantization
    centers_idx = reshape(qnt, (img.shape[0], img.shape[1]))
    clustered = centroids[centers_idx]
    figure(1)
    imshow(clustered)
    title("RGB quantization (k="+str(clusters)+")")
    savefig('Q2Figures/'+name+'_qztRGB.png')
    return clustered

#Part 2:
def quantizeHSV(img, clusters, name):
    #reshape the pixels matrix
    pixel = reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    #conver to hsv
    h=[]
    sv=[]
    for row in pixel:
        m=colorsys.rgb_to_hsv(row[0]/255.0, row[1]/255.0, row[2]/255.0)
        #m=colorsys.rgb_to_hsv(row[0], row[1], row[2])
        h.append(m[0])
        sv.append([m[1], m[2]])
    #pixelhsv = array(sv)
    pixelh=array(h)
    #perform clustering
    centroids,_ = kmeans(pixelh, clusters)

    #quantization on H-channel
    qnt,_ = vq(pixelh, centroids)
    qntizedH = centroids[qnt]

    #convert back to RGB
    clustered=[]
    for i in range(pixel.shape[0]):
        n=colorsys.hsv_to_rgb(qntizedH[i], sv[i][0], sv[i][1])
        clustered.append([n[0]*255.0, n[1]*255.0, n[2]*255.0])
        #if (img.shape[2] == 3):
        #    clustered.append([n[0]*255.0, n[1]*255.0, n[2]*255.0])
        #else:
        #    clustered.append([n[0], n[1], n[2], 1.0])
    res = array(clustered)
    res2 = reshape(res, (img.shape))

    figure(1)
    imshow(res2.astype(uint8))
    title("H-channel quantization (k="+str(clusters)+")")
    savefig('Q2Figures/'+name+'_qztH.png')
    return res2.astype(uint8)

#Part 3:
def getSSD(img1, img2):
    diff = img1-img2
    diff2 = diff*diff
    return diff2.sum()

#Part 4:
def histHChl(img, clusters, name):
    pixel = reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    #get H channel
    h=[]
    for row in pixel:
        m = colorsys.rgb_to_hsv(row[0]/255.0, row[1]/255.0, row[2]/255.0)
        h.append(m[0])

    plt.figure()
    plt.hist(h, bins=100)
    plt.title("H Channel before Quantization")
    plt.savefig('Q2Figures/'+name+'_histBeforeQtz.png')

    pixelh=array(h)
    #perform clustering
    centroids,_ = kmeans(pixelh, clusters)
    #quantization on H-channel
    qnt,_ = vq(pixelh, centroids)
    qntizedH = centroids[qnt]
    plt.figure()
    plt.hist(qntizedH, bins=100)
    plt.title("H Channel after Quantization")
    plt.savefig('Q2Figures/'+name+'_histAfterQtz.png')


#Part 5:
def q5(imgfile, clusters):
    img = imread(imgfile)
    if (img.shape[2] == 4):
        img = delete(img,3,2)*255.0
    name = imgfile[8]+'_'+str(clusters)
    resRGB = quantizeRGB(img, clusters, name)
    resHSV = quantizeHSV(img, clusters, name)
    ssd1 = getSSD(img, resRGB)
    ssd2 = getSSD(img, resHSV)
    print "SSD of RGB quantization: ", ssd1
    print "SSD of H-channel quantization: ", ssd2
    histHChl(img, clusters, name)


if __name__ == '__main__':
    if not os.path.exists("Q2Figures"):
       os.makedirs("Q2Figures")

    ks = {4, 6}
    images = {'colorful1.jpg', 'colorful2.jpg', 'colorful3.png'}
    for image in images:
        print "Image: ", image
        for k in ks:
            print "k = ", k
            q5(image, k)