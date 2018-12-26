import cv2 as cv
import numpy as np

def openImage(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return image

def filterMatchesByDistance(matches):
    filteredMatches = []
    matches = sorted(matches, key = lambda x:x.distance)
    ptsPairs = min(400, len(matches)*0.3)
    filteredMatches = matches[:ptsPairs]
    return filteredMatches

def filterMatchesRANSAC(matches, keypointsA, keypointsB):
    filteredMatches = []
    if len(matches) >= 4:
        src_pts = np.float32([ keypointsA[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypointsB[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)
        matchesMask = mask.ravel().tolist()

        for i in range(len(matchesMask)):
            if matchesMask[i] == 1:
                filteredMatches.append(matches[i])

    return homography, filteredMatches

def showResult(imgA, keypointsA, imgB, keypointsB, matches, homography=None):
    imgMatch = cv.drawMatches(imgA, keypointsA, imgB, keypointsB, matches, None)

    if homography is not None:
        h,w = imgA.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, homography)
        dst += np.float32([w,0])
        imgMatch = cv.polylines(imgMatch,[np.int32(dst)],True,(0,255,0),3, cv.LINE_AA)

    cv.namedWindow("matches", cv.WINDOW_KEEPRATIO)
    cv.imshow("matches", imgMatch)
    cv.waitKey(1000)

def main():

    detector = cv.xfeatures2d.SIFT_create()
    matcher = cv.FlannBasedMatcher()
    
    ## POSTER TEST
    image_test = openImage('poster_test.jpg')
    # find the keypoints and descriptors    
    kp_test, des_test = detector.detectAndCompute(image_test,None)
    #img_test = cv.drawKeypoints(image_test, kp_test, None, color=(255,0,0))
    #cv.imshow('image FAST',img_test)
    #cv.waitKey(1000)
    
    name = str("poster")
    type_image = str(".jpg")

    for i in range (1,8):
        filename = name + str(i) + type_image
        image = openImage(filename)
        # find the keypoints and descriptors
        kp, des = detector.detectAndCompute(image,None)
        
        matches = matcher.match(des_test,des)
        MatchesDist = filterMatchesByDistance(matches)
        #showResult(image_test,kp_test, image, kp, MatchesDist)

        homoRansac , MatchesRansac = filterMatchesRANSAC(MatchesDist,kp_test,kp)        
        showResult(image_test,kp_test, image, kp, MatchesRansac, homoRansac)

if __name__ == '__main__':
    main()
