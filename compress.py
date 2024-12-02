from PIL import Image
import os

def resize_and_compress_image(input_path, output_path, new_size=(480, 270)):
    """Resize and compress images without numpy."""
    try:
        # Open the image file
        with Image.open(input_path) as img:
            # Resize image to the new size
            img_resized = img.resize(new_size, Image.LANCZOS)

            # Save the image with compression
            img_resized.save(output_path, format="JPEG", quality=85)
            print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_images(input_directory, output_directory):
    """Process all images in a directory."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('png', 'jpg', 'jpeg')):
            # Define output path for each image
            output_path = os.path.join(output_directory, filename)
            # Resize and compress the image
            resize_and_compress_image(input_path, output_path)

if __name__ == "__main__":
    input_directory = "./python_scripts/extracted_images"  # Modify with your input directory
    output_directory = "./python_scripts/reduced"  # Modify with your output directory
    process_images(input_directory, output_directory)
