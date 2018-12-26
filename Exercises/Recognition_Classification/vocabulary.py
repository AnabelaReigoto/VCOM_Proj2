import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

class Vocabulary:
    def __init__(self, nWords):
        self.vocabulary = None
        self.nWords = nWords
    def train(self, listOfImages):
        #detector = cv.xfeatures2d.SIFT_create()
        detector = cv.KAZE_create()
        allDescriptors = []
        for name in listOfImages:
            img = openImage(name)
            if img is None:
                continue
            keypoints, descriptors = detector.detectAndCompute(img, None)
            allDescriptors.extend(descriptors)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness,labels,centers = cv.kmeans(np.float32(allDescriptors), self.nWords, None, criteria, 100, cv.KMEANS_PP_CENTERS)
        self.vocabulary = centers

    def whichWord(self, descriptor):
        if self.vocabulary.shape[0] <= 1:
            return -1

        minIndex = 0
        #minDistance = cv.norm(self.vocabulary[0,:]-descriptor,cv.NORM_L2)
        minDistance = np.linalg.norm(self.vocabulary[0,:]-descriptor)

        for i in range(1, self.vocabulary.shape[0]):
            #distance = cv.norm(self.vocabulary[i,:]-descriptor,cv.NORM_L2)
            distance = np.linalg.norm(self.vocabulary[i,:]-descriptor)
            if distance < minDistance:
                minDistance = distance
                minIndex = i

        return minIndex

def openImage(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return image

def drawKeypoints(windowName, image, keypoints, words):
    if len(keypoints) != len(words):
        return

    newImage = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    maxw=0
    for word in words:
        if word>maxw:
            maxw = word

    steps = int(255/(math.log(maxw+1)/math.log(3)))
    colors = []
    for r in range(1,256,steps):
        for g in range(1,256,steps):
            for b in range(1,256,steps):
                colors.append((b,g,r))

    positions = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
    for i in range(len(keypoints)):
        cv.circle(newImage, positions[i], 4, (colors[words[i]]), 2)

    cv.namedWindow( windowName, cv.WINDOW_AUTOSIZE )
    cv.imshow( windowName, newImage )

def main():
    ##################################################
    #detector = cv.xfeatures2d.SIFT_create()
    detector = cv.KAZE_create()
    ##################################################

    name = str("poster")
    type_image = str(".jpg")
    listImages = [7]
    for i in range (1,8):
        name = name + str(i) + type_image
        listImages.append(name)
        #print(listImages[i])

    listimg = ['poster1.jpg','poster2.jpg', 'poster3.jpg', 'poster4.jpg','poster5.jpg','poster6.jpg','poster7.jpg']
    voc = Vocabulary(10)
    trained_voc = voc.train(listimg)

    ## POSTER TEST
    image_test = openImage('poster_test.jpg')
    # find the keypoints and descriptors    
    kp_test, des_test = detector.detectAndCompute(image_test,None)
    
    corresp_words = []
    for n in des_test:
        index = voc.whichWord(n)
        corresp_words.append(index)

    

    img_test = cv.drawKeypoints(image_test, kp_test, None, color=(255,0,0))
    cv.imshow('Keypoints PosterTest',img_test)
    cv.waitKey(1000)





if __name__ == '__main__':
    main()