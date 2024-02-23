# width, height, top, left

import pytesseract

import numpy as np
import cv2
import sys
import time

if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def formatImageOCR(originalScreenshot):
    screenshot = np.array(originalScreenshot, dtype=np.uint8)
    grayscaleImage = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(grayscaleImage, 245, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.bitwise_not(binary)
    kernel = np.ones((2, 2), np.uint8)
    final_image = cv2.dilate(binary, kernel, iterations=1) 
    return final_image


# Change to https://stackoverflow.com/questions/66334737/pytesseract-is-very-slow-for-real-time-ocr-any-way-to-optimise-my-code 
# or https://www.reddit.com/r/learnpython/comments/kt5zzw/how_to_speed_up_pytesseract_ocr_processing/

def getTextFromImage(image):
    """ returns text from image """
    imageCandidate = formatImageOCR(image)
    # Write result to disk:
    
    # DEBUG log round to disk
    # import time
    # cv2.imwrite(f"./DEBUG/{str(time.time())}.png", imageCandidate, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # NOTE: This part seems to be buggy
    # Get current round from screenshot with tesseract
    return pytesseract.image_to_string(imageCandidate,  config='--psm 10 --oem 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/').replace("\n", ""), imageCandidate

