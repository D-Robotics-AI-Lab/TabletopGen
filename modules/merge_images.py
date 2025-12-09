import os
import cv2
import numpy as np
import math
from pathlib import Path
import argparse

def get_image_files(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file in os.listdir(folder_path):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)

    def extract_number(filename):
        stem = Path(filename).stem
        try:
            last_underscore = stem.rfind('_')
            if last_underscore != -1:
                number_str = stem[last_underscore + 1:]
                return int(number_str)
        except ValueError:
            pass
        return float('inf')
    
    image_files.sort(key=extract_number)
    return image_files

def calculate_grid_size(num_images):
    """Calculate the optimal grid size (rows and columns)"""
    if num_images == 0:
        return 0, 0
    
    # Calculate a grid close to a square
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    return rows, cols

def resize_image_keep_aspect(image, target_width, target_height):
    """Resize image to target size while keeping aspect ratio"""
    h, w = image.shape[:2]
    
    # Calculate scaling factors
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)
    
    # Calculate new size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create a canvas with target size (white background)
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    
    # Calculate center position
    start_y = (target_height - new_h) // 2
    start_x = (target_width - new_w) // 2
    
    # Place the resized image at the center of the canvas
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return canvas

def add_text_to_image(image, text, position='bottom'):
    """Add text label to the image"""
    h, w = image.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 0, 0)
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Fixed text area height
    text_area_height = 30
    result_image = np.ones((h + text_area_height, w, 3), dtype=np.uint8) * 255
    
    result_image[:h, :] = image
    
    # Calculate text position (centered)
    text_x = (w - text_width) // 2
    text_y = h + text_height + 5
    
    # Add text
    cv2.putText(result_image, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return result_image

def merge_images(folder_path, output_name="all_images"):
    """Merge all images in a folder into one large image"""
    try:
        image_files = get_image_files(folder_path)
        
        if not image_files:
            print(f"No image files found in folder {folder_path}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Calculate grid size
        rows, cols = calculate_grid_size(len(image_files))
        
        images = []
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img, Path(img_file).stem)) 
            else:
                print(f"Warning: Unable to read image {img_file}")
        
        if not images:
            print("No images were successfully read")
            return
        
        # Set target size for each cell
        cell_width = 300
        cell_height = 300
        text_area_height = 30
        
        # Create the final large canvas
        final_width = cols * cell_width
        final_height = rows * (cell_height + text_area_height)
        final_image = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
        
        # Place each image into the grid
        for idx, (img, img_name) in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            # Resize image to cell size (excluding text area)
            resized_img = resize_image_keep_aspect(img, cell_width, cell_height)
            
            # Add text label
            img_with_text = add_text_to_image(resized_img, img_name)
            
            # Ensure the generated image size is correct
            expected_height = cell_height + text_area_height
            if img_with_text.shape[0] != expected_height:
                print(f"Warning: Image {img_name} size mismatch, adjusting")
                corrected_img = np.ones((expected_height, cell_width, 3), dtype=np.uint8) * 255
                corrected_img[:cell_height, :] = resized_img
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                color = (0, 0, 0)
                (text_width, text_height), baseline = cv2.getTextSize(img_name, font, font_scale, thickness)
                text_x = (cell_width - text_width) // 2
                text_y = cell_height + text_height + 5
                cv2.putText(corrected_img, img_name, (text_x, text_y), font, font_scale, color, thickness)
                
                img_with_text = corrected_img
            
            # Calculate position in the final image
            start_y = row * (cell_height + text_area_height)
            start_x = col * cell_width
            end_y = start_y + cell_height + text_area_height
            end_x = start_x + cell_width
            
            # Place image
            final_image[start_y:end_y, start_x:end_x] = img_with_text
        
        output_path = os.path.join(folder_path, f"{output_name}.jpg")
        cv2.imwrite(output_path, final_image)
        
        print(f"Merge complete! Saved to: {output_path}")
        
    except Exception as e:
        print(f"Error merging images: {str(e)}")


if __name__ == "__main__":
    input_dir = 'output_scene/scene_1/output_assets/layout_rotation_images'
    output = "all_images"
    
    if not os.path.exists(input_dir):
        print(f"Error: Folder {input_dir} does not exist")
    
    merge_images(input_dir, output)

