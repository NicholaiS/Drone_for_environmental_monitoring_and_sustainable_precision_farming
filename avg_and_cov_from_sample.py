import cv2
import numpy as np

def find_avg_and_cov():
    img = cv2.resize(cv2.imread("Drone pics\\20230811_115843151_iOS.jpg"), (1080, 720))
    pixels = np.reshape(img, (-1, 3))

    # Finding the mean of the color and the covariance:
    img_annotated = cv2.resize(cv2.imread('Drone pics\\20230811_115843151_iOS_rod.jpg'), (1080, 720))
    mask = cv2.inRange(img_annotated, (30, 45, 230), (40, 55, 235))
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