import cv2
import pytesseract
import numpy as np
import pandas as pd

# Constants
CONFIG_PSM = '--psm 8'
LOCATION_LOG_COLUMNS = ['Plate Number', 'Location']

class LocationLog:
    def __init__(self):
        self.log = pd.DataFrame(columns=LOCATION_LOG_COLUMNS)

    def append(self, plate_number, location):
        self.log = self.log.append({'Plate Number': plate_number, 'Location': location}, ignore_index=True)

class ALPR:
    def __init__(self):
        self.location_log = LocationLog()

    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return binary

    def perform_ocr(self, image):
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image)
        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(preprocessed_image, config=CONFIG_PSM)
        return text.strip()

    def track_location(self, plate_number, location):
        self.location_log.append(plate_number, location)

    def capture_image(self):
        # Prompt the user to enter the image file path
        image_path = input("Enter the path to the image file (e.g., C:\\path\\to\\image.jpg): ")
        image = cv2.imread(image_path)
        return image

    def main(self):
        # Capture an image
        image = self.capture_image()
        if image is None:
            print(f"Error: Image '{image_path}' not found.")
            return

        # Perform OCR to extract the license plate number
        plate_number = self.perform_ocr(image)
        print(f"Detected Plate Number: {plate_number}")

        # Track the location of the vehicle
        self.track_location(plate_number, 'Downtown')

        # Print the location log
        print("\nLocation Log:")
        print(self.location_log.log)

if __name__ == "__main__":
    alpr = ALPR()
    alpr.main()
