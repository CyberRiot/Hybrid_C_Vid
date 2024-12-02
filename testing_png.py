import os

# Directory containing PNG files
image_dir = "./python_scripts/reduced"
output_file = "VOOD\data\binary\output_binary_file.data"

with open(output_file, 'wb') as output:
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".png"):
            with open(os.path.join(image_dir, filename), 'rb') as image_file:
                output.write(image_file.read())
                output.write(b'\n')  # Add newline if needed
print(f"Binary file {output_file} created successfully.")
