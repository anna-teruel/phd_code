"""
Raw video preprocessing before running the analysis pipeline.
@author Anna Teruel-Sanchis, 2023
"""

import cv2
import numpy as np
import os 
import subprocess

def process_video(input_video, output_dir):
    """
    Process a video file by creating a minimum intensity projection and subtracting it from each frame.

    Args:
        input_video (str): input video file path
        output_dir (str): directory where the output video will be saved

    Raises:
        ValueError: Error opening video file
    """
    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Prepare to write the processed video
    filename = os.path.basename(input_video)
    processed_filename = f"p_{filename}"
    output_video_path = os.path.join(output_dir, processed_filename)  # Correct full path including filename
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # Use the full path here

    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()

    cap.release()
    frames = np.array(frames)

    min_intensity_proj = np.min(frames, axis=0)
    subtracted_frames = frames - min_intensity_proj

    # Write the processed frames to the video file
    for frame in subtracted_frames:
        out.write(frame.astype(np.uint8))

    out.release()
    print(f"Processed video saved as {output_video_path}")


def process_directory(root_dir):
    """
    Process all videos in a directory by creating a minimum intensity projection and subtracting it from each frame.

    Args:
        root_dir (str): root directory containing the videos to process
    """    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.avi') and not file.startswith('p_'):
                input_video_path = os.path.join(root, file)
                output_video_path = os.path.join(root, f"p_{file}")
                process_video(input_video_path, output_video_path)

def convert_to_mjpeg(input_video, output_dir):
    """
    Convert a video file to MJPEG AVI format using FFmpeg.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str): The path where the output video will be saved.
    """
    command = [
        'ffmpeg',
        '-i', input_video,
        '-c:v', 'mjpeg',
        '-q:v', '3',
        output_dir
    ]
    subprocess.run(command, check=True)
