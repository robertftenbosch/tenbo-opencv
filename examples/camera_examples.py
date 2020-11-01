"""
Example to introduce how to read a camera connected to your computer
"""

# Import the required packages
import cv2
import argparse
from skimage.filters import (threshold_otsu, threshold_triangle, threshold_niblack, threshold_sauvola)
from skimage import img_as_ubyte
import numpy as np
def sketch_image(img):
    """Sketches the image applying a laplacian operator to detect the edges"""

    # Convert to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median filter
    img_gray = cv2.medianBlur(img_gray, 5)

    # Detect edges using cv2.Laplacian()
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)

    # Threshold the edges image:
    ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

    return thresholded


def cartonize_image(img, gray_mode=False):
    """Cartoonizes the image applying cv2.bilateralFilter()"""

    # Get the sketch:
    thresholded = sketch_image(img)

    # Apply bilateral filter with "big numbers" to get the cartoonized effect:
    filtered = cv2.bilateralFilter(img, 10, 1200, 1200)

    # Perform 'bitwise and' with the thresholded img as mask in order to set these values to the output
    cartoonized = cv2.bitwise_and(filtered, filtered, mask=thresholded)

    if gray_mode:
        return cv2.cvtColor(cartoonized, cv2.COLOR_BGR2GRAY)

    return cartoonized

def outline_image(image):
    outline_kernel = np.array([[-1, -2, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])



    return cv2.filter2D(image, -1, outline_kernel)
def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def sepia_two(image):
    img = np.array(image, dtype=np.float64) # converting to float to prevent loss
    img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img[np.where(img > 255)] = 255 # normalizing values greater than 255 to 255
    img = np.array(img, dtype=np.uint8)
    return img
def pencil(image):
    dst_gray, dst_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.07)
    return dst_gray
# We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()

# We add 'index_camera' argument using add_argument() including a help.
parser.add_argument("index_camera", help="index of the camera to read from", type=int)
args = parser.parse_args()
print(args.index_camera)
# We create a VideoCapture object to read from the camera (pass 0):
capture = cv2.VideoCapture(args.index_camera)

# Get some properties of VideoCapture (frame width, frame height and frames per second (fps)):
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# Print these values:
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(frame_width))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(frame_height))
print("CAP_PROP_FPS : '{}'".format(fps))

# Check if camera opened successfully
if capture.isOpened()is False:
    print("Error opening the camera")
 
# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame from the camera
    ret, frame = capture.read()

    if ret is True:
        # Display the captured frame:

        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
        cartoon_image = cartonize_image(frame)
        cv2.imshow('Input cartoon_image from the camera', cartoon_image)
        pencil_image = pencil(frame)
        cv2.imshow('Input pencil_image from the camera', pencil_image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100,200)
        cv2.imshow('Input edges from the camera', edges)

        sketch_image_image = sketch_image(frame)
        cv2.imshow('Input sketch_image_image from the camera', sketch_image_image)
        height, width = cartoon_image.shape[:2]
        temp = cv2.resize(cartoon_image, (64, 64), interpolation=cv2.INTER_LINEAR)
        pixel_cartoon = cv2.resize(temp, (width,height), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Input pixel_cartoon from the camera', pixel_cartoon)
        # Trying Sauvola's scikit-image algorithm:
        # thresh_sauvola = threshold_sauvola(cartoon_image, window_size=25)
        # binary_sauvola = cartoon_image > thresh_sauvola
        # binary_sauvola = img_as_ubyte(binary_sauvola)
        # # Convert the frame captured from the camera to grayscale:
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('input binary_sauvola', binary_sauvola)
        # # Display the grayscale frame:
        cv2.imshow('input camera', frame)

        # cartoon_frame = outline_image(frame)
        # bitwise_xor = cv2.bitwise_and(frame, cartoon_frame)
        # # Display the grayscale frame:
        # cv2.imshow('cartoon_frame input camera', bitwise_xor)
 
        # Press q on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# Release everything:
capture.release()
cv2.destroyAllWindows()
