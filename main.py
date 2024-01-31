import numpy as np
import cv2
from skimage import color, segmentation
from skimage.filters import gabor, gabor_kernel
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import slic


def extract_pixel_features(image):
    rgb_feature = image.reshape((-1, 3))
    rgb_feature_normalized = rgb_feature / 255.0
    h, w, _ = image.shape
    spatial_feature = np.column_stack((rgb_feature, np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))

    spatial_feature_normalized = spatial_feature.astype(np.float64)

    spatial_feature_normalized[:, :3] /= 255.0

    spatial_feature_normalized[:, -2:] /= np.array([h - 1, w - 1])
    np.set_printoptions(suppress=True)

    return np.array(rgb_feature_normalized), np.array(spatial_feature_normalized)


def extract_superpixel_features(image, n_segments):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segments = segmentation.slic(image, n_segments, compactness=10.0)
    rgb_mean_superpixel = image

    for segment_id, unique_segment_id in enumerate(np.unique(segments)):
        mask = (segments == unique_segment_id)
        mean_rgb = np.mean(rgb_mean_superpixel[mask])
        rgb_mean_superpixel[mask] = mean_rgb

    height, width, channels = rgb_mean_superpixel.shape
    rgb_mean_superpixel = rgb_mean_superpixel.reshape((height * width, channels))
    print(rgb_mean_superpixel.shape)
    print(np.array(rgb_mean_superpixel))

    return np.array(rgb_mean_superpixel)


def histogram(image, n_segments, n_clusters):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segments = segmentation.slic(image_rgb, n_segments=n_segments, compactness=10)

    superpixel_histograms = []

    for segment_id in np.unique(segments):
        mask = (segments == segment_id)
        superpixel_rgb = image_rgb[mask]
        hist_r, _ = np.histogram(superpixel_rgb[:, 0], bins=256, range=(0, 256), density=True)
        hist_g, _ = np.histogram(superpixel_rgb[:, 1], bins=256, range=(0, 256), density=True)
        hist_b, _ = np.histogram(superpixel_rgb[:, 2], bins=256, range=(0, 256), density=True)
        superpixel_histogram = np.concatenate([hist_r, hist_g, hist_b])
        superpixel_histograms.append(superpixel_histogram)

    print(np.array(superpixel_histograms).shape)


    labels, centroids = k_means(np.array(superpixel_histograms), n_clusters)
    before_cluster = labels

    cluster_assignments = np.zeros_like(segments)
    for i, segment_id in enumerate(np.unique(segments)):
        mask = (segments == segment_id)
        cluster_assignments[mask] = labels[i]

    return cluster_assignments, before_cluster


def superpixel_gabor(image, path, segment):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = segmentation.slic(image_rgb, n_segments=segment, compactness=10)

    gabor_array = gabor_features(image, path)
    image_array = image_rgb



    for seg in np.unique(segments):
        mask = (segments == seg)
        mean_gabor = np.mean(gabor_array[mask])

        image_array[mask] = mean_gabor


    height, width, channels = image_array.shape

    rgb_gabor_superpixel = image_array.reshape((height * width, channels))
    print("gabır array:", np.array(rgb_gabor_superpixel))

    print(rgb_gabor_superpixel.size)

    return np.array(rgb_gabor_superpixel)


def gabor_features(image, path):

    freq = 0.5
    tetas = [0, 45, 90, 180]
    gabor_responses = []
    kernel_array = []

    for teta in tetas:
        kernel = np.real(gabor_kernel(frequency=freq, theta=np.deg2rad(teta)))
        kernel_array.append(kernel)

    image_path = path
    image = cv2.imread(image_path)
    for ker in kernel_array:
        new_img = np.zeros(image.shape)
        new_img[:, :, 0] = ndi.convolve(image[:, :, 0], ker, mode='wrap')
        new_img[:, :, 1] = ndi.convolve(image[:, :, 1], ker, mode='wrap')
        new_img[:, :, 2] = ndi.convolve(image[:, :, 2], ker, mode='wrap')
        gabor_responses.append(new_img)

    empty_image_mean = np.zeros(image.shape)

    for matrix in gabor_responses:
        empty_image_mean += matrix

    empty_image_mean /= len(gabor_responses)
    empty_image_mean = np.array(empty_image_mean)

    return np.array(empty_image_mean)


def k_means(data, k, max_iterations=100, tol=1e-4):

    centroids = data[np.random.choice(len(data), k, replace=True)]

    for iteration in range(max_iterations):

        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)


        new_centroids = np.array([np.mean(data[labels == i], axis=0) for i in range(k)])


        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids
    return labels, centroids

def mapping(labels):
    unique_labels = np.unique(labels)
    label_to_number = {}
    for number, label in enumerate(unique_labels):
        label_to_number[label] = number

    numbers = np.zeros_like(labels)
    for i, label in enumerate(labels):
        numbers[i] = label_to_number[label]

    return numbers



def visualize_results():

    # the way i print the image segments.
    fig, axes = plt.subplots(1, 1, figsize=(50, 50))
    image = cv2.imread("Your_image_path")
    superpixel_segments = segmentation.slic(image, n_segments=500, compactness=10)
    mean = extract_superpixel_features(image, 500)
    labels, centroids = k_means(mean, k=3)
    label_numbers = mapping(labels)
    label_numbers_2d = label_numbers.reshape(superpixel_segments.shape)
    axes.imshow(segmentation.mark_boundaries(image, label_numbers_2d))
    plt.show()

    # the way i print the superpixels and pixel features.
    #for Histogram:
    '''hist = histogram(image, 500, 3)
    axes[0, 0].imshow(hist.reshape(image.shape[:-1]), cmap='viridis')'''
    #for gabor:
    ''' gabor = superpixel_gabor(image,"your_image_path" ,500)
    gabor_clustered = k_means(gabor, 4)[0]
    axes[0, 0].imshow(gabor_clustered.reshape(image.shape[:-1]), cmap='viridis')'''
    #to print the superpixels:
    '''segments = slic(image, n_segments=100, compactness=10)
    axes[1, 2].imshow(segmentation.mark_boundaries(image, segments))'''

    fig, axes = plt.subplots(5, 2, figsize=(50, 50))

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[0, 0].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[0, 0].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[0, 1].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[0, 1].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[1, 0].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[1, 0].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[1, 1].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[1, 1].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[2, 0].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[2, 0].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[2, 1].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[2, 1].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[3, 0].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[3, 0].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[3, 1].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[3, 1].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[4, 0].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[4, 0].set_title("Pixel-level RGB Clustering")

    image = cv2.imread("Your_image_path")
    mean = extract_superpixel_features(image, 500)
    superpixel_labels_rgb = k_means(mean, 4)[0]
    axes[4, 1].imshow(superpixel_labels_rgb.reshape(image.shape[:-1]), cmap='viridis')
    axes[4, 1].set_title("Pixel-level RGB Clustering")

    plt.show()
    plt.close('all')


visualize_results()


