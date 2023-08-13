import cv2
import numpy as np
import matplotlib.pyplot as plt

indir = 'C:\\Users\\Nicho\\OneDrive\\Dokumenter\\VSC Workspace\\Visuals for drone (Summer course)\\Drone pics\\'
outdir = 'C:\\Users\\Nicho\\OneDrive\\Dokumenter\\VSC Workspace\\Visuals for drone (Summer course)\\Drone pics out\\'

# Reading the image and resizing:
img = cv2.resize(cv2.imread(indir + "20230811_115843151_iOS.jpg"), (1080, 720))
height, width = img.shape[:2]
#img = cv2.resize(img, (1080, 720))
pixels = np.reshape(img, (-1, 3))

# ------------------------------------------- Color segmentation ---------------------------------------------
# Farven du vil ramme i RGB: (157, 95, 38)
img_annotated = cv2.resize(cv2.imread(indir + '20230811_115843151_iOS_rod.jpg'), (1080, 720))
mask = cv2.inRange(img_annotated, (245, 0, 0), (256, 10, 10))
mask_pixels = np.reshape(mask, (-1))
cv2.imshow("Image", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Determine mean value, standard deviations and covariance matrix
# for the annotated pixels.
# Using cv2 to calculate mean and standard deviations
mean, std = cv2.meanStdDev(img, mask = mask)

pixels = np.reshape(img, (-1, 3))
mask_pixels = np.reshape(mask, (-1))
annot_pix_values = pixels[mask_pixels == 255, ]
avg = np.average(annot_pix_values, axis=0)
cov = np.cov(annot_pix_values.transpose()) 

print("Mean color values of the annotated pixels")
print(avg)
print("Covariance matrix of color values of the annotated pixels")
print(cov)

# ------------------------------------------- Feature Extraction ---------------------------------------------
