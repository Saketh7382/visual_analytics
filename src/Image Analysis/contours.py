import cv2 
import numpy as np
import random as rng
rng.seed(12345)

class SideLot:
    
    def __init__(self,image=2):
        src='/home/saketh/Documents/Hackathons/AI-ML Tractor Analytics/SideLotImages/Scrub_Store_{}.JPG'.format(image)
        self.src = src
        self.read_img()
        self.get_black()
    
    def read_img(self):
        self.image = cv2.imread(self.src,0)
        cv2.waitKey(0)
        #self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.median_blur = cv2.medianBlur(self.image,3)
        self.source = self.median_blur
        
    def get_black(self):
        c = np.uint8(0)
        a = np.array([[np.array([c,c,c])]*self.image.shape[1]]*self.image.shape[0])
        self.black = np.array(a)

    def findEdges(self,thresh_1=100,thresh_2=200,show=False):
        self.edged = cv2.Canny(self.source, thresh_1, thresh_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()       

        self.contours, self.hierarchy = cv2.findContours(self.edged, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        print("Number of Contours found = " + str(len(self.contours)))
        
        if show:
            cv2.imshow('Canny Edges After Contouring', self.edged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return self.contours, self.hierarchy

images = 5
for img in range(1,images+1):
    analyser = SideLot(img)
    print(analyser.image.shape)
    contours, hierarchy = analyser.findEdges(100,700)
    black = analyser.black
    
    long_c = list()
    min_thresh = 50
    
    for i in contours:
        if i.shape[0] >= min_thresh:
            long_c.append(i)
    
    for i in long_c:
        a,b,c,d = i[0][0][0], i[0][0][1], i[-1][0][0], i[-1][0][1]
        cv2.drawContours(black, np.array([[[a,b]]]), 0, (255,255,255), 3)
        cv2.drawContours(black, np.array([[[c,d]]]), 0, (255,255,255), 3)
    
    for i in range(len(long_c)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(black, long_c, i, color, 1, cv2.LINE_8, hierarchy, 0)
    
    #cv2.imshow('Source', analyser.image)
      
    cv2.imshow('Contours', black)
    cv2.waitKey(0)
    cv2.destroyAllWindows()