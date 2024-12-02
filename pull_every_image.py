import os
import subprocess

def extract_all_frames(video_path, output_dir):
    """
    Extracts every frame from a video using ffmpeg.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory to save frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        os.path.join(output_dir, "frame_%06d.png")
    ]
    
    print(f"Extracting frames from {video_path}...")
    subprocess.run(ffmpeg_command, check=True)
    print(f"Frames saved to {output_dir}")

# Example usage
video_path = "input_video.mp4"  # Replace with your video file path
output_dir = "output_frames"   # Replace with your desired output directory
extract_all_frames(video_path, output_dir)