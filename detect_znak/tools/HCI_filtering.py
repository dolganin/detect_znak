import cv2 
import numpy as np

def Xor_img(img, mask):
    
    mask[mask != 0] = 255
    res = np.bitwise_and(img, mask)
    return res

if __name__ == "__main__":
    img = cv2.imread("./datasets/HCI/frame_1711441002149887687.jpg")
    mask = cv2.imread("./detect_znak/doroga.jpg")
    masked = Xor_img(img)
    # cv2.imshow("test", masked)
    # cv2.waitKey(0)
