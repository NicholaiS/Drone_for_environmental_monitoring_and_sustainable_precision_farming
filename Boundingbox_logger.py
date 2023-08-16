from pickle import FALSE
import cv2
import numpy as np
import platform
import os

pic_amount = 23
CIRCULARITY = 0.80

folder = 'Bounding boxes out\\' + str(CIRCULARITY) + "\\"
if not os.path.exists(folder):
    os.makedirs(folder)

for i in range(pic_amount):
    # Reading the image and resizing:
    img = cv2.resize(cv2.imread("Drone pics\\Outdoor test 1\\img_" + str(i) + ".jpg"), (1080, 720))
    height, width = img.shape[:2]
    pixels = np.reshape(img, (-1, 3))

    # ------------------------------------------- Color segmentation ---------------------------------------------

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
    
    # Saving unscaled and scaled images.
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

    with open(folder + "img_" + str(i) + ".txt", 'w') as f:
        for contour in contours:
            # Calculate circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0: circularity = (4 * np.pi * area) / (perimeter ** 2)
            print("Circularity:", circularity)
    
            if circularity > CIRCULARITY:
                # Calculate moments for contour
                moments = cv2.moments(contour)

                drawing = 0 * imgfeGray
                # Calculate centroid
                if moments['m00'] != 0:  # Avoid division by zero
                    cX = int(moments['m10'] / moments['m00'])
                    cY = int(moments['m01'] / moments['m00'])
                    centroid = (cX, cY)

                    # Estimate the radius based on the contour area
                    estimated_radius = np.sqrt(area / np.pi)
        
                    # Calculate bounding box coordinates
                    top_left = (int(cX - estimated_radius), int(cY - estimated_radius))
                    bottom_right = (int(cX + estimated_radius), int(cY + estimated_radius))
        
                    # Draw the bounding square
                    bounding_square_color = (0, 0, 255)  # BGR color for the square (red)
                    img_maha_scaled = cv2.rectangle(img_maha_scaled, top_left, bottom_right, bounding_square_color, 1)
                    f.write(f"{top_left} {bottom_right}\n")    


    cv2.imshow("Scaled image with Bounding Boxes", img_maha_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
