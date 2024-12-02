import os

def save_as_binary(input_directory, output_file):
    """Convert image files to binary and save to a single file."""
    with open(output_file, "wb") as f_out:
        for filename in os.listdir(input_directory):
            input_path = os.path.join(input_directory, filename)
            if os.path.isfile(input_path) and filename.lower().endswith(('png', 'jpg', 'jpeg')):
                with open(input_path, "rb") as f_in:
                    f_out.write(f_in.read())  # Read and write the image file as binary
                print(f"Converted {filename} to binary.")

if __name__ == "__main__":
    input_directory = "./python_scripts/reduced"  # Modify with your resized images directory
    output_file = "./VOOD/data/binary/output_binary_file.data"  # Modify with the desired output file name
    save_as_binary(input_directory, output_file)