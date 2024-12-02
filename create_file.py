import os
import cv2
import numpy as np
import struct

def write_image_to_binary(image_path, binary_file):
    """
    Write an image to a binary file with metadata.
    
    Args:
    - image_path: Path to the image file
    - binary_file: Open file object for the binary file
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    
    if image is None:
        print(f"Error: Failed to load image {image_path}")
        return
    
    # Get image metadata
    height, width = image.shape
    image_size = image.nbytes  # Size in bytes
    
    # Write metadata (header)
    binary_file.write(struct.pack('I', width))  # Write width (4 bytes)
    binary_file.write(struct.pack('I', height))  # Write height (4 bytes)
    binary_file.write(struct.pack('I', image_size))  # Write size (4 bytes)

    # Write the raw image data
    binary_file.write(image.tobytes())  # Write the image data as raw bytes
    print(f"Processed and added {image_path} to the binary file.")

def create_binary_file(image_dir, output_binary_file):
    """
    Create a binary file with images and their metadata.
    
    Args:
    - image_dir: Directory containing the image files
    - output_binary_file: Path to the binary output file
    """
    with open(output_binary_file, 'wb') as bin_file:
        # Iterate over all image files in the directory
        for image_name in os.listdir(image_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, image_name)
                write_image_to_binary(image_path, bin_file)
        print(f"All images from {image_dir} have been processed and added to {output_binary_file}.")

# Example usage
image_directory = "E:/Code/python_scripts/reduced"  # Directory containing your images
binary_file_path = "E:/Code/VOOD/data/binary/output_binary_file.data"  # Output binary file

create_binary_file(image_directory, binary_file_path)
