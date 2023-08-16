from pickle import FALSE
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

loadColor = False

image_number = 0

# Der skal logges Hu Moments og andet relevant information, så vi kan se om det er andet vi kan bruge...
folder = 'Info logs\\'

# Reading the image and resizing:
img = cv2.resize(cv2.imread("Drone pics\\Outdoor test 1\\img_0.jpg"), (1080, 720))
height, width = img.shape[:2]
pixels = np.reshape(img, (-1, 3))
cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def find_avg_and_cov():
    # Finding the mean of the color and the covariance:
    img_annotated = cv2.resize(cv2.imread('Drone pics\\Outdoor test 1\\img_9_rod.jpg'), (1080, 720))
    mask = cv2.inRange(img_annotated, (0, 0, 245), (10, 10, 255))
    mask_pixels = np.reshape(mask, (-1))
    cv2.imshow("Mask Image", mask)
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

    # Save avg and cov to a text file
    np.savetxt("avg_cov.txt", np.concatenate((avg, cov.flatten())))

# ------------------------------------------- Color segmentation ---------------------------------------------

if loadColor:
   find_avg_and_cov()

# Finding Mahalanobis distance using the mean and covariance:
# Load avg and cov from the text file
loaded_data = np.loadtxt("avg_cov.txt")
loaded_avg = loaded_data[:3]
loaded_cov = loaded_data[3:].reshape(3, 3)

reference_color = loaded_avg # Given in BGR
covariance_matrix = loaded_cov

# Calculate the euclidean distance to the reference_color annotated color.
shape = pixels.shape
diff = pixels - np.repeat([reference_color], shape[0], axis=0)
inv_cov = np.linalg.inv(covariance_matrix)
moddotproduct = diff * (diff @ inv_cov)
mahalanobis_dist = np.sum(moddotproduct, axis=1)
mahalanobis_distance_image = np.reshape(mahalanobis_dist, (img.shape[0], img.shape[1]))

# Scale the distance image and export it.
mahalanobis_distance_image_scaled = 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
cv2.imshow("Mahalanobis Distance Image", mahalanobis_distance_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Pics out/mahalanobis_dist_image.jpg", mahalanobis_distance_image)
cv2.imwrite("Pics out/mahalanobis_dist_image_scaled.jpg", mahalanobis_distance_image_scaled)

# ------------------------------------------- Feature Extraction ---------------------------------------------
img_maha = cv2.imread("Pics out/mahalanobis_dist_image.jpg")
img_maha_scaled = cv2.imread("Pics out/mahalanobis_dist_image_scaled.jpg")
imgfeGray = cv2.cvtColor(img_maha, cv2.COLOR_BGR2GRAY)
tLower = 50
tUpper = 250
imgCanny = cv2.Canny(imgfeGray, tLower, tUpper)

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_maha_scaled, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contour Image", img_maha_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
with open(folder + "img_" + str(i) + ".txt", 'w') as f:
    for contour in contours:
        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0: circularity = (4 * np.pi * area) / (perimeter ** 2)
        print("Circularity:", circularity)
    
        if circularity > 0.8:
            # Calculate moments for contour
            moments = cv2.moments(contour)

            # Calculate centroid
            if moments['m00'] != 0:  # Avoid division by zero
                cX = int(moments['m10'] / moments['m00'])
                cY = int(moments['m01'] / moments['m00'])
                centroid = (cX, cY)
        
                # Draw centroid on the image
                cv2.circle(img_maha_scaled, centroid, 5, (0, 0, 255), -1)

                # Calculate Hu Moments
                hu_moments = cv2.HuMoments(moments)
                print("Hu Moments:", hu_moments.flatten())
    


cv2.imshow("Contour Image with Centroids", img_maha_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
