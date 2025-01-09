# This script is to process the data for training the model

# make the image size to have the same height and the width

from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import shutil
IMAGE_EXTENSION = ['.jpg', '.jpeg', '.png']

def make_image_height_width_same(image: np.ndarray, target_size: int = None) -> np.ndarray:
    """Make the image height and width equal and resize to power of 2 if needed.
    
    Args:
        image: Input image as numpy array
        target_size: Target size for the output image (must be power of 2)
        
    Returns:
        Processed square image with dimensions being power of 2
    """
    height, width = image.shape[:2]
    
    # First make the image square
    size = max(height, width)
    square_img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Calculate padding
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    
    # Place the original image in the center
    square_img[y_offset:y_offset+height, x_offset:x_offset+width] = image
    
    # If target_size is not specified, find the nearest power of 2
    if target_size is None:
        target_size = 2 ** (size - 1).bit_length()  # Find nearest power of 2
    
    # Resize to target size
    if size != target_size:
        square_img = cv2.resize(square_img, (target_size, target_size), 
                              interpolation=cv2.INTER_LANCZOS4)
    
    return square_img

def process_data(data_dir: Path, output_dir: Path, target_size: int = 512):
    """Process all images in data_dir and save to output_dir.
    
    Args:
        data_dir: Input directory containing images
        output_dir: Output directory to save processed images
        target_size: Target size for the output images (must be power of 2)
    """
    # Verify target_size is a power of 2
    if not (target_size & (target_size - 1) == 0):
        raise ValueError("target_size must be a power of 2")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # json data to copy
    source_json_path = data_dir / 'dataset.json'
    target_json_path = output_dir / 'dataset.json'
    shutil.copy(source_json_path, target_json_path)

    # Get all image files
    image_files = []
    for ext in IMAGE_EXTENSION:
        image_files.extend(data_dir.glob(f'*{ext}'))
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make image square and resize to power of 2
        processed_image = make_image_height_width_same(image, target_size)
        
        # Convert back to BGR for saving
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        
        # Save processed image
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), processed_image)

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--target_size', type=int, default=512, 
                      help='Target size for output images (must be power of 2)')
    args = parser.parse_args()
    process_data(args.data_dir, args.output_dir, args.target_size)

if __name__ == '__main__':
    main()


