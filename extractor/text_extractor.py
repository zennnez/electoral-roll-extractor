"""
Text extraction utilities for electoral roll extraction.

This file contains functions for extracting text from processed images using OCR.
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import logging
from typing import Dict, Any, Tuple, List

from config import IMAGE_PARAMS
from extractor.image_processor import remove_watermark

# Set up logger
logger = logging.getLogger(__name__)

def extract_number(
    img: np.ndarray, 
    x: int, 
    y: int, 
    w: int, 
    h: int
) -> str:
    """
    Extract a number from a specific region in the image, enhancing contrast and 
    using OCR to recognize the digits.

    Args:
        img: The input image in BGR format.
        x: The x-coordinate of the top-left corner of the region.
        y: The y-coordinate of the top-left corner of the region.
        w: The width of the region.
        h: The height of the region.

    Returns:
        The extracted number as a string.
    """
    try:
        # Extract the region of interest
        roi = img[y:y+h, x:x+w]
        
        # Convert to PIL image for enhancement
        pil_img = Image.fromarray(roi)
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(IMAGE_PARAMS['contrast_enhancement'])
        
        # Convert back to numpy array
        roi = np.array(pil_img)
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Create binary image
        _, binary = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find non-white columns
        col_sums = np.sum(binary, axis=0)
        
        # Check if there are any non-white pixels
        if np.max(col_sums) > 0:
            # Find the rightmost non-white column
            rightmost_col = np.max(np.where(col_sums > 0))
            
            # Extract a small region around the rightmost number
            number_width = IMAGE_PARAMS['number_width']
            number_roi = roi[0:h, max(0, rightmost_col-number_width):rightmost_col+5]
            
            # Perform OCR with specific configuration for digits
            number = pytesseract.image_to_string(
                number_roi, 
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
            ).strip()
            
            return number
        
        return ""
    except Exception as e:
        logger.error(f"Error extracting number: {e}")
        return ""


def extract_text(
    img: np.ndarray, 
    x: int, 
    y: int, 
    w: int, 
    h: int
) -> str:
    """
    Extract text from a specific region in the image using OCR.

    Args:
        img: The input image in BGR format.
        x: The x-coordinate of the top-left corner of the region.
        y: The y-coordinate of the top-left corner of the region.
        w: The width of the region.
        h: The height of the region.

    Returns:
        The extracted text as a string.
    """
    try:
        # Extract the region of interest
        roi = img[y:y+h, x:x+w]
        
        # Perform OCR with page segmentation mode 6 (assume a single uniform block of text)
        text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


def fix_page_ocr_numbers(page_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix OCR numbers in a page by ensuring they form a sequential series.
    
    Args:
        page_data: List of voter data dictionaries for a single page
        
    Returns:
        List of voter data with corrected OCR numbers
    """
    if not page_data:
        return page_data
        
    # Sort by box position (they should already be sorted)
    page_data = sorted(page_data, key=lambda x: x['box'])
    
    # Get all OCR numbers that look valid
    valid_numbers = []
    for entry in page_data:
        try:
            num = int(entry['ocr_number'])
            valid_numbers.append(num)
        except (ValueError, TypeError):
            continue
            
    if not valid_numbers:
        return page_data
        
    # Find the most common difference between consecutive numbers
    diffs = [valid_numbers[i+1] - valid_numbers[i] for i in range(len(valid_numbers)-1)]
    if not diffs:
        return page_data
        
    # Most common difference should be 1
    expected_diff = 1
    
    # Start with the first valid number
    expected_number = valid_numbers[0]
    
    # Fix the series
    for entry in page_data:
        entry['number'] = str(expected_number)
        expected_number += expected_diff
        
    return page_data

def process_voter_box(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    inner_box: Tuple[int, int, int, int],
    current_box_index: int,
    page_number: int,
    prev_box_data: Dict[str, Any] = None,
    use_ocr_numbers: bool = False
) -> Dict[str, Any]:
    """
    Process a voter information box and extract all relevant text fields.
    Returns None if the box appears to be fake/empty.
    """
    try:
        # Extract coordinates
        x, y, w, h = box
        ix, iy, iw, ih = inner_box
        
        # Extract the box image
        box_img = img[y:y+h, x:x+w]
        
        # Remove watermark
        clean_img = remove_watermark(box_img)
        
        # Extract voter number from inner box
        ocr_number = extract_number(clean_img, ix, iy, iw, ih)
        
        # Extract top right text (EPIC number)
        top_right_text = extract_text(clean_img, iw+10, 0, w-iw-10, ih)
        
        # Define region for the main text block
        text_x = 5
        text_y = iy + ih + 5
        text_w = int(w * 2/3)
        text_h = h - text_y - 5
        
        # Extract main text
        text = extract_text(clean_img, text_x, text_y, text_w, text_h)
        
        # Split into lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # Check if this is likely a fake/empty box
        total_text = ' '.join([ocr_number, top_right_text] + lines)
        if len(total_text.strip()) < 5:  # Box has almost no text
            return None
            
        # Determine number based on mode
        if use_ocr_numbers:
            number = ocr_number if ocr_number else ''
        else:
            # Calculate expected number based on previous box or box index
            if prev_box_data and 'number' in prev_box_data and prev_box_data['number']:
                try:
                    expected_number = str(int(prev_box_data['number']) + 1)
                except ValueError:
                    expected_number = str(current_box_index + 1)
            else:
                expected_number = str(current_box_index + 1)
            number = expected_number
        
        # Ensure we have exactly 4 lines
        while len(lines) < 4:
            lines.append('')
        
        # Take only the first 4 lines
        lines = lines[:4]
        
        # Return structured data
        return {
            'number': number,
            'ocr_number': ocr_number,
            'top_right_text': top_right_text,
            'line1': lines[0],
            'line2': lines[1],
            'line3': lines[2],
            'line4': lines[3]
        }
    except Exception as e:
        logger.error(f"Error processing voter box: {e}")
        return None  # Return None instead of empty dict for failed boxes
