import os

# Define paths
input_file_path = './VOOD/data/binary/output_binary_file.data'  # Path to your binary data file
output_dir = './python_scripts/test_images.py'

# PNG file signature for identifying PNG files in binary data
png_signature = b'\x89PNG\r\n\x1a\n'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def extract_png_images(binary_file):
    with open(binary_file, 'rb') as f:
        data = f.read()

    # List to store where PNG files start and end
    png_start_indices = []

    # Find all occurrences of the PNG signature in the binary file
    idx = 0
    while idx < len(data):
        idx = data.find(png_signature, idx)
        if idx == -1:
            break
        png_start_indices.append(idx)
        print(f'Found PNG signature at index: {idx}')  # Debugging info
        idx += len(png_signature)

    # Check if any PNGs were found
    if not png_start_indices:
        print("No PNG signatures found.")
        return

    # Extract each PNG and handle possible newline separators
    for i in range(len(png_start_indices)):
        start = png_start_indices[i]
        if i + 1 < len(png_start_indices):
            end = png_start_indices[i + 1]
        else:
            end = len(data)

        # Extract the PNG data, ensuring newlines are skipped
        png_data = data[start:end]  # No changes here, we will check raw data

        # Save each PNG file
        output_file_path = os.path.join(output_dir, f'image_{i + 1}.png')

        with open(output_file_path, 'wb') as out_file:
            out_file.write(png_data)

        print(f'Extracted: {output_file_path}')


def inspect_binary_file(file_path, num_bytes=512):
    with open(file_path, 'rb') as f:
        data = f.read(num_bytes)
    print(data)

# Inspect the first 512 bytes of the binary file
inspect_binary_file(input_file_path)
# Run the extraction
extract_png_images(input_file_path)
