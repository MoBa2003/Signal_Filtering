import cv2
import numpy as np


filters = {
    # فیلترهای شناسایی لبه
    "Sobel X": np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]),
    "Sobel Y": np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]]),
    "Prewitt X": np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]]),
    "Prewitt Y": np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]]),
    "Laplacian": np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]]),
    "Laplacian of Gaussian (LoG)": np.array([[ 0,  0, -1,  0,  0],
                                             [ 0, -1, -2, -1,  0],
                                             [-1, -2, 16, -2, -1],
                                             [ 0, -1, -2, -1,  0],
                                             [ 0,  0, -1,  0,  0]]),
    "Diagonal" : np.array([
        [1,0,-1],
        [0,0,0],
        [-1,0,1]
    ]),

    # فیلترهای محوشدگی
    "Box Blur": np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]) / 9,
    "Gaussian Blur": np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16,

    # فیلترهای برجسته‌سازی
    "Sharpen": np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]]),
    "Edge Enhancement": np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]]),

    # فیلترهای خاص
    "Emboss": np.array([[-2, -1,  0],
                        [-1,  1,  1],
                        [ 0,  1,  2]]),
    "Outline": np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]])
}


# تعریف تابع برای اعمال کانولوشن روی هر کانال رنگی
def apply_convolution_per_channel(image, kernel):
    
    filtered_image = cv2.filter2D(image, -1, filters[kernel])
    return filtered_image

# بارگذاری تصویر
img = cv2.imread("./q3/img3.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Original", img)


filter = "Outline"
# اعمال کانولوشن با کرنل Sobel
edge_image = apply_convolution_per_channel(img,filter)

# نمایش نتیجه
cv2.imshow(f'Edges Detected ({filter})', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
