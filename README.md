##Image Segmentation and Feature Extraction##
This Python script performs image segmentation and feature extraction on images using various techniques, including superpixel clustering, Gabor filtering, and k-means clustering. The script is organized into different functions, each serving a specific purpose.

#Requirements#
Python 3.x
NumPy
OpenCV
scikit-image
Matplotlib
SciPy

#Usage#
You should change the following line image = cv2.imread("Your_image_path") with the image path in you pc.
And it is given only 1 calling nd printing example in the visualize_result method for image labals, you can call the other superpixel features according to that example. For printing the Histogram result it is enough to put the return value of histogram method to the cv2.imshow method.

#Functions#
1. extract_pixel_features(image)
Extracts pixel features from the input image.
Returns normalized RGB features and spatial features.
2. extract_superpixel_features(image, n_segments)
Performs superpixel segmentation on the input image.
Computes the mean RGB value for each superpixel.
Returns a 1D array of superpixel features.
3. histogram(image, n_segments, n_clusters)
Computes histograms of RGB values for each superpixel.
Applies k-means clustering to the histograms.
Returns cluster assignments and labels before clustering.
4. superpixel_gabor(image, path, segment)
Applies Gabor filtering to the input image.
Computes the mean Gabor response for each superpixel.
Returns a 1D array of superpixel features.
5. gabor_features(image, path)
Applies Gabor filtering to the input image.
Returns an array of Gabor responses.
6. k_means(data, k, max_iterations=100, tol=1e-4)
Applies k-means clustering to input data.
Returns cluster labels and centroids.
7. mapping(labels)
Maps cluster labels to numeric values.
8. visualize_results()
Visualizes the results of superpixel clustering and pixel-level RGB clustering on a sample image.

#Image Links#
Input images: https://drive.google.com/drive/folders/1z5Id0kmEX4sNl8zjOW3NhMu0fLRmPc4W?usp=drive_link

Superpixels:
100- https://drive.google.com/drive/folders/1OJMU9D9DNP-yGxAQCZ_h2WcIrApc03OR?usp=drive_link
500- https://drive.google.com/drive/folders/1_ggL8b95AaD9uBKf4v2SZ2D8n7Ezb5LW?usp=drive_link
1000- https://drive.google.com/drive/folders/1iR-8_-zuwlAMs81nD5dB2WuoCJ3T986x?usp=drive_link

Labels:
RGB color- https://drive.google.com/drive/folders/1v07U9gkXb1R1DMNHev5xfa_EQ6UWW4RX?usp=drive_link
RGB spatial- https://drive.google.com/drive/folders/1DStHxyuE4OmdAfRtHtd9LfB2g5V_D2YI?usp=drive_link
Mean superpixel- https://drive.google.com/drive/folders/1MNV8XUF0mXm_Vd8oplRd8690Q032PESb?usp=drive_link
Histogram superpixel- https://drive.google.com/drive/folders/1nVTBbPLeD4dys97TmY_0J-vQp5cff-uJ?usp=drive_link
Gabor superpixel- https://drive.google.com/drive/folders/1HeYnfh3hnIdwaHctyzWjJuD3LwwVDYFn?usp=drive_link

Image segments:
https://drive.google.com/drive/folders/1UtTk87L-XxGx-u7Xtj0yRFsik7glL4iX?usp=drive_link
