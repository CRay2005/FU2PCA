import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description='Stage-1')
parser.add_argument('-i', '--image', help='input image', required=True)
args = parser.parse_args()

iter_cnt = 10
image = cv2.imread(args.image)
rect = (1, 1, image.shape[0]-2, image.shape[1]-2)
mask = np.zeros(image.shape[:2], dtype="uint8")

# apply GrabCut using the the bounding box segmentation method
(mask, _, _) = cv2.grabCut(image, mask, rect, np.zeros((1, 65), dtype="float"), np.zeros((1, 65), dtype="float"), iterCount=iter_cnt, mode=cv2.GC_INIT_WITH_RECT)

# the output mask has four possible output values, marking each pixel in the mask as: 
# (1) cv2.GC_BGD: definite background, 
# (2) cv2.GC_PR_BGD: probable background,,
# (3) cv2.GC_FGD: definite foreground, 
# (4) cv2.GC_PR_FGD: probable foreground
# set all definite background and probable background to 0
outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
# scale the mask from the range [0, 1] to [0, 255]
outputMask = (outputMask * 255).astype("uint8")

# ellipse the mask to get a smaller mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)) #ksize=5,5
smaller_mask = cv2.erode(outputMask, kernel, iterations=3)
cv2.imwrite("grabcut-smask.jpg", smaller_mask)
cv2.imwrite("grabcut-image.jpg", image)
cv2.imwrite("grabcut-omask.jpg", outputMask)

border = cv2.bitwise_xor(outputMask, smaller_mask)
print(f"{border.shape}")
cv2.imwrite("grabcut-border.jpg", border)

