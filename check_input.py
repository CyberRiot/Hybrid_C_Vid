import os
import cv2
import numpy as np
import struct

def check_image_integrity(image_path):
    """
    Check if an image file exists, can be loaded, and is not empty.
    
    Args:
    - image_path: Path to the image file (e.g., .png or .jpg)
    
    Returns:
    - True if the image is valid and not empty, False otherwise.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist!")
        return False
    
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Reading the image in grayscale
    if image is None:
        print(f"Error: Failed to load the image from '{image_path}'.")
        return False
    
    # Check if the image is empty
    if np.sum(image) == 0:  # If the sum of all pixel values is zero, the image is all black (empty)
        print(f"Error: The image '{image_path}' is empty (all black).")
        return False
    
    # Check if the feature vector is non-empty
    feature_vector = image.flatten()
    if feature_vector.size == 0:
        print(f"Error: Feature vector is empty for '{image_path}'.")
        return False
    
    print(f"Image '{image_path}' is valid and its feature vector is non-empty.")
    return True

def check_binary_integrity(binary_path):
    """
    Check the integrity of the binary file containing the image data.
    
    Args:
    - binary_path: Path to the binary file containing the images.
    
    Returns:
    - True if the binary file can be read and processed successfully, False otherwise.
    """
    if not os.path.exists(binary_path):
        print(f"Error: The binary file '{binary_path}' does not exist!")
        return False
    
    try:
        with open(binary_path, 'rb') as bin_file:
            # Read the binary file in chunks and verify its contents
            while True:
                # Here we assume the binary file contains images in a specific format
                # For example, each image might have a fixed header (e.g., size info) followed by the image data
                # The following is an example of how you might read a fixed-size image from the binary file
                # Update the size and structure to match your actual binary format
                image_size = struct.unpack('I', bin_file.read(4))[0]  # Read the size of the image data (4 bytes)
                image_data = bin_file.read(image_size)  # Read the image data
                
                if len(image_data) != image_size:
                    print(f"Error: Inconsistent image size in binary file. Expected {image_size} bytes, got {len(image_data)}.")
                    return False
                
                # Convert the binary image data into an image (assuming it's stored in grayscale format)
                image = np.frombuffer(image_data, dtype=np.uint8)
                if image.size == 0:
                    print(f"Error: Empty image data found in binary file.")
                    return False
                
                # Check the image integrity
                feature_vector = image.flatten()
                if feature_vector.size == 0:
                    print(f"Error: Feature vector is empty for an image in the binary file.")
                    return False
                
                # Optionally, you could also check the contents of the image if needed (e.g., check for empty images)
                if np.sum(image) == 0:
                    print(f"Error: Empty (all-black) image found in binary file.")
                    return False
                
                print(f"Successfully read and verified image from binary file.")
                break  # Break after checking one image for simplicity, extend if you want to check all images
    except Exception as e:
        print(f"Error: Failed to read binary file due to: {e}")
        return False
    
    return True

def check_directory_for_images(directory_path):
    """
    Check all image files in the specified directory for integrity.
    
    Args:
    - directory_path: Path to the directory containing the image files.
    """
    # List all files in the directory
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist!")
        return
    
    # Get all image files in the directory (we assume .png and .jpg files, but you can adjust as needed)
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in the directory '{directory_path}'.")
        return
    
    # Check each image for integrity
    for image_file in image_files:
        if check_image_integrity(image_file):
            print(f"Proceeding with processing of '{image_file}'.")
        else:
            print(f"Skipping '{image_file}' due to errors.")

def process_files(directory_path, binary_file):
    """
    Process all image files in a directory and check the binary file for integrity.
    
    Args:
    - directory_path: Directory containing the image files
    - binary_file: Path to the binary file containing the images
    """
    # Check all images in the directory
    check_directory_for_images(directory_path)
    
    # Check the integrity of the binary file
    if check_binary_integrity(binary_file):
        print(f"Binary file '{binary_file}' is valid. Proceeding with further processing.")
    else:
        print(f"Error: The binary file '{binary_file}' is corrupted or invalid.")

def generate_binary_from_images(image_directory, binary_file_path):
    """
    Generate a binary file from the images in the specified directory.
    
    Args:
    - image_directory: Path to the directory containing the image files
    - binary_file_path: Path where the binary file will be saved
    """
    # Open the binary file for writing
    with open(binary_file_path, 'wb') as bin_file:
        for image_file in os.listdir(image_directory):
            image_path = os.path.join(image_directory, image_file)
            
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process image files
                try:
                    # Convert the image to a feature vector
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
                    feature_vector = image.flatten()
                    
                    # Write the feature vector size (4 bytes) followed by the actual feature vector data
                    bin_file.write(struct.pack('I', len(feature_vector)))  # Write size of the feature vector
                    bin_file.write(np.array(feature_vector, dtype=np.uint8).tobytes())  # Write the actual vector data
                    print(f"Processed and added '{image_file}' to the binary file.")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")

# Example usage
if __name__ == "__main__":
    # Directory containing the image files
    image_directory = "E:\\Code\\python_scripts\\reduced"
    
    # Path to your binary file containing the images
    binary_file_path = "E:\\Code\\VOOD\\data\\binary\\output_binary_file.data"
    
    # First, generate the binary file from images
    generate_binary_from_images(image_directory, binary_file_path)
    
    # Process the files
    process_files(image_directory, binary_file_path)
