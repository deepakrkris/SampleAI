import numpy as np
import cv2
import easyocr

def convert_image_to_grayscale(file_name = "18_1.jpg", base = 'source/') :
  image = cv2.imread(base + file_name)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

  inverted_image = 255 - binary_image

  image = inverted_image

  return image

def remove_characters(image) :
  # print("begin initialize ocr")
  # Initialize the PaddleOCR reader
  reader = easyocr.Reader(['en'])

  # Use EasyOCR to detect text
  results = reader.readtext(image)
  #print("ocr detection completed")

  # Create a mask for the text
  mask = np.zeros_like(image)

  # Loop through the results and create rectangles in the mask
  for (bbox, text, prob) in results:
    # Extract the bounding box coordinates
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = (int(top_left[0]), int(top_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
    
    # Draw a rectangle on the mask
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)

  print("mask rectangles completed")
  
  # Inpaint the image using the mask
  image = cv2.inpaint(image, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

  return image