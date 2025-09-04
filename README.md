# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** Jeeva K 
- **Register Number:** 212223230090

  ### Ex. No. 01

#### 1. Read the image ('lion.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("lion.jpg")
bgr_img=cv2.imread("lion.jpg",0)
```

#### 2. Print the image width, height & Channel.
```python
bgr_img.shape
bgr_img.size
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img)
plt.imshow(bgr_img)
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('act.png',img_bgr)
png_image=cv2.imread('act.png')
plt.imshow(png_image[:,:,::-1])
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
color_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(color_img)
plt.title('Color Image')
plt.axis('on')
plt.show()
```

#### 7. Crop the image to extract any specific (Lion alone) object from the image.
```python
cropped_img=s_color_img[25:400,50:400]
plt.imshow(cropped_img[:,:,::-1])
plt.title('Cropped image')
plt.axis('off')
plt.show()
```

#### 8. Resize the image up by a factor of 2x.
```python
resized_img = cv2.resize(cropped_img, None, fx=50, fy=50, interpolation=cv2.INTER_LINEAR)
resized_img.shape
```

#### 9. Flip the cropped/resized image horizontally.
```python
flipped_img = cv2.flip(cropped_img, 1)
plt.imshow(flipped_img[:,:,::-1])
plt.title('Flipped Image')
plt.axis('off')
plt.show()
```

#### 10. Read in the image ('').
```python

```

#### 11. Add the following text to the dark area at the bottom of the image :
```python
image_text = cropped_img.copy()

text = 'Jeeva K '
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_color = white
font_thickness = 2

image_text = cv2.putText(image_text, text, (200, 325), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA)

plt.imshow(image_text[:, :, ::-1])
```

#### 12. Draw a magenta rectangle for lion.
```python
yellow  = (0, 255, 255)
red     = (0, 0, 255)
magenta = (255, 0, 255)
green   = (0, 255, 0)
white = (255,255,255)
image_rectangle = cropped_img.copy()
image_rectangle = cv2.rectangle(image_rectangle, (0,0), (350, 375), magenta, thickness = 10, lineType = cv2.LINE_8)
plt.imshow(image_rectangle[:, :, ::-1])
```

#### 13. Display the final annotated image.
```python
# YOUR CODE HERE
```

#### 14. Read the image ('act.jpg').
```python
# YOUR CODE HERE
```

#### 15. Adjust the brightness of the image.
```python
# Create a matrix of ones (with data type float64)
# matrix_ones = 
# YOUR CODE HERE
```

#### 16. Create brighter and darker images.
```python
matrix = np.ones(cropped_img.shape, dtype = 'uint8') * 30

img_brighter = cv2.add(cropped_img, matrix)
img_darker   = cv2.subtract(cropped_img, matrix)


```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize = [18,5])
plt.subplot(131); plt.imshow(img_darker[:, :, ::-1]);   plt.title('Darker')
plt.subplot(132); plt.imshow(cropped_img[:, :, ::-1]);          plt.title('Original')
plt.subplot(133); plt.imshow(img_brighter[:, :, ::-1]); plt.title('Brighter')
```

#### 18. Modify the image contrast.
```python
matrix1 = np.ones(cropped_img.shape) * 0.8
matrix2 = np.ones(cropped_img.shape) * 1.2

img_lower  = np.uint8(cv2.multiply(np.float64(cropped_img), matrix1))
img_higher = np.uint8(cv2.multiply(np.float64(cropped_img), matrix2))
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize = [18,5])
plt.subplot(131); plt.imshow(img_lower[:, :, ::-1]);  plt.title('Lower Contrast')
plt.subplot(132); plt.imshow(cropped_img[:, :, ::-1]);        plt.title('Original')
plt.subplot(133); plt.imshow(img_higher[:, :, ::-1]); plt.title('Higher Contrast')
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
img_bgr = cv2.imread('actress.jpg', cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_bgr)
plt.figure(figsize = [20, 10])
plt.subplot(141); plt.imshow(r); plt.title('Red Channel')
plt.subplot(142); plt.imshow(g); plt.title('Green Channel')
plt.subplot(143); plt.imshow(b); plt.title('Blue Channel')

```

#### 21. Merged the R, G, B , displays along with the original image
```python
imgMerged = cv2.merge((r, g, b))

plt.subplot(144)
plt.imshow(imgMerged)
plt.title('Merged Output')
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)
```
#### 23. Merged the H, S, V, displays along with original image.
```python
plt.figure(figsize = [20, 5])
plt.subplot(141); plt.imshow(h); plt.title('H Channel')
plt.subplot(142); plt.imshow(s); plt.title('S Channel')
plt.subplot(143); plt.imshow(v); plt.title('V Channel')

plt.subplot(144); plt.imshow(img_bgr[:, :, ::-1]); plt.title('Original')
```

## Output:
- **i)** Read and Display an Image. 
 
- **ii)** Adjust Image Brightness.  
- **iii)** Modify Image Contrast.  
- **iv)** Generate Third Image Using Bitwise Operations.

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

