import subprocess
import os

def extract_keyframes(input_video, output_folder):
    command = [
        'ffmpeg', '-i', input_video, '-vf', 'select=eq(pict_type\\,I)', '-fps_mode', 'vfr', f'{output_folder}\\frame_%04d.png'
    ]
    subprocess.run(command)

# Fix your file path
input_video = 'F:\\Code\\python_scripts\\video_files\\Polyphia.mp4'  # Ensure it's a raw string
output_folder = 'F:\\Code\\VOOD\\data\\'  # Ensure it's a valid directory

print(os.path.exists('F:\\Code\\python_scripts\\video_files\\Polyphia.mp4'))

# Extract keyframes
extract_keyframes(input_video, output_folder)
