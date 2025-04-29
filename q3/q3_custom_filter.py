import cv2
import numpy as np


def create_feature_based_filter(image, kernel_size=3):
    """
    This function analyzes the input image to detect important features like edges, corners, and intensity patterns,
    and creates a custom filter based on these features.

    :param image: Input image (color or grayscale)
    :param kernel_size: Size of the custom kernel (must be an odd number, default: 3)
    :return: Custom feature-based filter
    """
    # If the image is in color, convert it to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 1. Detect edges using the Canny edge detector
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    # 2. Detect corners using the Harris corner detector
    corners = cv2.cornerHarris(gray_image.astype(np.float32), blockSize=2, ksize=3, k=0.04)

    # Normalize corner response for better scaling
    corners = cv2.normalize(corners, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 3. Use the intensity gradient (Sobel filters) to capture texture information
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Combine edges, corners, and gradient magnitude into one feature map
    combined_features = edges.astype(np.float32) + corners + gradient_magnitude

    # 4. Create an empty kernel and fill it based on the feature map
    custom_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # Resize the combined feature map to match the kernel size
    resized_features_x = cv2.resize(np.sum(combined_features, axis=0).reshape(1, -1), (kernel_size, 1))
    resized_features_y = cv2.resize(np.sum(combined_features, axis=1).reshape(-1, 1), (1, kernel_size))

    # Fill the kernel's central row and column with the resized features
    center = kernel_size // 2
    custom_filter[center, :] = resized_features_x.flatten()
    custom_filter[:, center] = resized_features_y.flatten()

    # Normalize the kernel to have values in the range [-1, 1]
    custom_filter -= np.mean(custom_filter)
    custom_filter /= np.max(np.abs(custom_filter))

    return custom_filter


# Test the function
if __name__ == "__main__":
    # Load the input image
    img = cv2.imread("./q3/img1.webp", cv2.IMREAD_COLOR)

    # Generate a custom edge-based filter
    custom_filter = create_feature_based_filter(img, kernel_size=5)
    print("Custom Edge-Based Filter:\n", custom_filter)

    # Apply the custom filter to the input image using convolution
    filtered_image = cv2.filter2D(img, -1, custom_filter)

    # Display the original and filtered images
    cv2.imshow("Original Image", img)
    cv2.imshow("Filtered Image (Custom Filter)", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
