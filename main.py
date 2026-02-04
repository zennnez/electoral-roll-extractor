"""
Electoral Roll Data Extractor

This script extracts voter information from electoral roll PDFs, processes
the extracted data, and outputs structured information to CSV and Excel files.
It processes all PDF files in the input directory and creates corresponding output files with the same base filename.

Usage:
    python main.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--debug] [--numocr]
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pdf2image
import pytesseract
from tqdm import tqdm

# Import configuration
import config

# Import project modules
from extractor.image_processor import (
    preprocess_image, 
    find_boxes, 
    find_inner_boxes,
    create_debug_image
)
from extractor.text_extractor import process_voter_box, fix_page_ocr_numbers
from extractor.data_processor import process_raw_data

def setup_logging():
    """Configure logging for the application."""
    # Create handlers
    file_handler = logging.FileHandler(config.LOG_FILE)
    console_handler = logging.StreamHandler()
    
    # Set log levels
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def configure_environment():
    """Configure the environment for OCR and image processing."""
    logger = logging.getLogger(__name__)
    
    # Set Tesseract path
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    logger.info(f"Tesseract path set to: {config.TESSERACT_CMD}")
    
    # Verify PDF directory exists
    if not Path(config.INPUT_DIR).exists():
        logger.error(f"PDF directory not found: {config.INPUT_DIR}")
        raise FileNotFoundError(f"PDF directory not found: {config.INPUT_DIR}")
    
    # Verify Poppler path exists (if provided)
    if config.POPPLER_PATH and not Path(config.POPPLER_PATH).exists():
        logger.warning(f"Poppler path not found: {config.POPPLER_PATH}")
    
    logger.info("Environment configured successfully")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Electoral Roll Data Extractor")
    
    parser.add_argument(
        "--input-dir", 
        type=str,
        default=str(config.INPUT_DIR),
        help="Directory containing electoral roll PDF files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with additional output and visualizations"
    )
    
    parser.add_argument(
        "--numocr",
        action="store_true",
        help="Use OCR numbers instead of sequential numbering"
    )
    
    return parser.parse_args()


def get_pdf_files(input_dir):
    """
    Get all PDF files from the input directory.
    
    Args:
        input_dir: Path to the directory containing PDF files.
        
    Returns:
        List of PDF file paths.
    """
    logger = logging.getLogger(__name__)
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return []
    
    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    return pdf_files


def generate_output_paths(pdf_path, output_dir):
    """
    Generate output file paths based on the PDF filename.
    
    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save output files.
        
    Returns:
        Tuple of (csv_path, excel_path).
    """
    pdf_name = Path(pdf_path).stem
    output_path = Path(output_dir)
    
    csv_path = output_path / f"{pdf_name}.csv"
    excel_path = output_path / f"{pdf_name}.xlsx"
    
    return csv_path, excel_path

def extract_data_from_pdf(pdf_path, use_ocr_numbers=False):
    """
    Extract voter information from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        use_ocr_numbers: Whether to use OCR numbers instead of sequential numbering
        
    Returns:
        A list of dictionaries containing extracted voter information.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting data extraction from: {pdf_path}")
    logger.info(f"Using {'OCR' if use_ocr_numbers else 'sequential'} numbers")
    
    # Convert PDF to images
    logger.info("Converting PDF to images...")
    images = pdf2image.convert_from_path(
        pdf_path, 
        poppler_path=config.POPPLER_PATH
    )
    logger.info(f"PDF converted to {len(images)} images")
    
    # Container for extracted data
    all_data = []
    prev_box_data = None
    total_boxes = 0
    valid_boxes = 0
    
    # Process each page
    for page_idx, image in enumerate(tqdm(images, desc="Processing pages")):
        # Convert PIL image to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess the image
        preprocessed = preprocess_image(img)
        
        # Find voter boxes in the page
        boxes = find_boxes(preprocessed)
        total_boxes += len(boxes)
        logger.info(f"Found {len(boxes)} potential voter boxes on page {page_idx+1}")
        
        # Sort boxes from top to bottom, left to right
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        # Save debug image of all boxes
        if config.IMAGE_PARAMS["save_debug_images"]:
            debug_img = create_debug_image(img, boxes)
            debug_path = os.path.join(config.DEBUG_DIR, f"page_{page_idx+1}_boxes.png")
            cv2.imwrite(debug_path, debug_img)
        
        # Container for page data
        page_data = []
        page_valid_boxes = 0
        
        # Process each box
        for box_idx, box in enumerate(tqdm(boxes, desc=f"Page {page_idx+1} boxes", leave=False)):
            x, y, w, h = box
            
            # Extract region of interest
            roi = preprocessed[y:y+h, x:x+w]
            
            # Find inner box (voter number box)
            inner_boxes = find_inner_boxes(roi)
            
            if inner_boxes:
                inner_box = inner_boxes[0]
                
                # Extract data from the box with additional parameters
                box_data = process_voter_box(
                    img=img, 
                    box=box, 
                    inner_box=inner_box,
                    current_box_index=box_idx,
                    page_number=page_idx + 1,
                    prev_box_data=prev_box_data,
                    use_ocr_numbers=use_ocr_numbers
                )
                
                # Only process valid boxes
                if box_data is not None:
                    # Add metadata
                    box_data["page"] = page_idx + 1
                    box_data["box"] = box_idx + 1
                    
                    # Add to page data
                    page_data.append(box_data)
                    prev_box_data = box_data
                    page_valid_boxes += 1
                    valid_boxes += 1
                    
                    # Save debug image at intervals
                    if (config.IMAGE_PARAMS["save_debug_images"] and 
                        box_idx % config.IMAGE_PARAMS["debug_image_interval"] == 0):
                        
                        debug_img = create_debug_image(img, [box], 0, inner_box)
                        debug_path = os.path.join(
                            config.DEBUG_DIR, 
                            f"page_{page_idx+1}_box_{box_idx+1}.png"
                        )
                        cv2.imwrite(debug_path, debug_img)
        
        # Fix OCR numbers for the entire page if using OCR mode
        if use_ocr_numbers and page_data:
            page_data = fix_page_ocr_numbers(page_data)
        
        # Add page data to all data
        all_data.extend(page_data)
        
        logger.info(f"Processed {page_valid_boxes} valid voter boxes out of {len(boxes)} potential boxes on page {page_idx+1}")
    
    logger.info(f"Extraction complete. Found {valid_boxes} valid voter entries out of {total_boxes} potential boxes across {len(images)} pages")
    return all_data


def save_raw_data(data, csv_path):
    """
    Save raw extracted data to a CSV file.
    
    Args:
        data: List of dictionaries containing extracted data.
        csv_path: Path to save the CSV file.
        
    Returns:
        The DataFrame containing the raw data.
    """
    logger = logging.getLogger(__name__)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw data saved to: {csv_path}")
    
    return df


def process_and_save_output(df, excel_path):
    """
    Process raw data and save structured output to Excel.
    
    Args:
        df: DataFrame containing raw extracted data.
        excel_path: Path to save the Excel file.
        
    Returns:
        The processed DataFrame.
    """
    logger = logging.getLogger(__name__)
    
    # Process the raw data
    processed_df = process_raw_data(df, config.INPUT_COLUMNS)
    
    # Select output columns
    output_df = processed_df[config.OUTPUT_COLUMNS]
    
    # Save to Excel
    output_df.to_excel(excel_path, index=False)
    logger.info(f"Processed data saved to: {excel_path}")
    
    return output_df


def main():
    """Main execution function."""
    # Set up logging
    logger = setup_logging()
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure environment
        configure_environment()
        
        # Get all PDF files from input directory
        pdf_files = get_pdf_files(args.input_dir)
        
        if not pdf_files:
            logger.error("No PDF files found in the input directory")
            return 1
        
        # Process each PDF file
        total_processed = 0
        successful_files = []
        failed_files = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing file: {pdf_file.name}")
                
                # Generate output paths for this file
                csv_path, excel_path = generate_output_paths(pdf_file, args.output_dir)
                
                # Extract data from PDF
                data = extract_data_from_pdf(pdf_file, use_ocr_numbers=args.numocr)
                
                if not data:
                    logger.warning(f"No data extracted from {pdf_file.name}")
                    failed_files.append(pdf_file.name)
                    continue
                
                # Save raw data to CSV
                raw_df = save_raw_data(data, csv_path)
                
                # Process and save output
                output_df = process_and_save_output(raw_df, excel_path)
                
                # Log success for this file
                logger.info(f"Successfully processed {pdf_file.name}: {len(output_df)} voter entries")
                successful_files.append(pdf_file.name)
                total_processed += len(output_df)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}", exc_info=True)
                failed_files.append(pdf_file.name)
                continue
        
        # Print final summary
        logger.info("=" * 60)
        logger.info(f"Electoral Roll Data Extraction Complete")
        logger.info(f"Total PDF files found: {len(pdf_files)}")
        logger.info(f"Successfully processed: {len(successful_files)}")
        logger.info(f"Failed to process: {len(failed_files)}")
        logger.info(f"Total voter entries processed: {total_processed}")
        
        if successful_files:
            logger.info(f"Successfully processed files: {', '.join(successful_files)}")
        
        if failed_files:
            logger.warning(f"Failed to process files: {', '.join(failed_files)}")
        
        logger.info(f"Output files saved to: {args.output_dir}")
        logger.info("=" * 60)
        
        return 0 if not failed_files else 1
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
