"""
Raw video preprocessing functions for DeepLabCut data.
@author Anna Teruel-Sanchis, 2024
"""
import cv2
import os
import csv
import subprocess

def extract_frame_from_videos(video_folder:str, output_format='jpeg', frame_time='00:00:05'):
    """
    Extract a frame from each video in the specified folder and save it as an image.

    This function processes each video in the given folder, extracting a frame from the
    specified time (default is the 5th second). The extracted frame is saved as an image
    in the specified format (default is 'jpeg').

    Args:
        video_folder (str): The path to the folder containing the videos.
        output_format (str, optional): The format in which to save the extracted frame. Defaults to 'jpeg'.
        frame_time (str, optional): The time in the video from which to extract the frame (format: 'HH:MM:SS'). 
            Defaults to '00:00:05'.
    """
    for file_name in os.listdir(video_folder):
        if file_name.endswith('.mp4'):
            video_path = os.path.join(video_folder, file_name)
            output_image_path = os.path.join(video_folder, f"{file_name[:-4]}.{output_format}")

            ffmpeg_command = [
                'ffmpeg', '-y',  
                '-i', video_path,  
                '-ss', frame_time, 
                '-vframes', '1',  
                output_image_path  
            ]

            subprocess.run(ffmpeg_command)
            print(f"Extracted frame from {file_name} and saved it as {output_image_path}")


def draw_rois(image_folder:str, output_csv:str):
    """
    Manually select Regions of Interest (ROIs) from a set of images and save them to a CSV file.

    The function processes all images in the given folder and allows the user to interactively
    select ROIs for each image. The ROI coordinates (x, y, width, height) are stored in a CSV file
    in the given directory.
    This function can only read "jpeg" images. Other formats will make the function fail. 

    Args:
        image_folder (str): Path to the folder containing images from which ROIs will be selected. 
            The function processes only images with '.jpeg', but this can be modified as needed.
        
        output_csv (str): Path to the CSV file where the selected ROI coordinates will be saved. 
            The file will include a row for each image with the columns: image name, x-coordinate, 
            y-coordinate, width, and height of the ROI.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "x", "y", "w", "h"])  

        for image_name in os.listdir(image_folder):
            if image_name.endswith('.jpeg'):  
                image_path = os.path.join(image_folder, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to load image: {image_name}")
                    continue

                roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
                cv2.destroyAllWindows()  # Close OpenCV windows
                
                x, y, w, h = roi

                writer.writerow([image_name, x, y, w, h])
                print(f"Saved ROI for {image_name}: x={x}, y={y}, w={w}, h={h}")


def process_videos_based_on_roi(video_folder, roi_csv):
    """
    Crop, scale, and convert videos to grayscale based on the Regions of Interest (ROIs) stored in a CSV file.

    The function processes videos in the specified folder based on the ROI coordinates provided in the CSV file.
    It applies cropping, scaling, and grayscale conversion to each video. Processed videos are saved with a modified name.

    This function will only process "mp4" files. Other formats will raise an error. 

    Args:
        video_folder (str): Path to the folder containing videos to be processed. Each video should correspond to an image
            in the CSV file.
        
        roi_csv (str): Path to the CSV file containing the ROI coordinates for each image. The CSV should have columns:
            image, x (x-coordinate), y (y-coordinate), w (width), and h (height).
    """
    with open(roi_csv, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_name = row['image']
            x = int(row['x'])
            y = int(row['y'])
            w = int(row['w'])
            h = int(row['h'])

            video_name = image_name.replace(".jpeg", ".mp4").replace(".jpg", ".mp4").replace(".png", ".mp4")
            video_path = os.path.join(video_folder, video_name)
            output_video_path = os.path.join(video_folder, video_name.replace(".mp4", "_cropped_scaled_gray.mp4"))

            if os.path.exists(video_path):
                ffmpeg_command = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-vf', f'crop={w}:{h}:{x}:{y},scale=iw/2:ih/2,format=gray',
                    '-c:a', 'copy', output_video_path
                ]
                subprocess.run(ffmpeg_command)
                print(f"Cropped, scaled, and converted {video_name} to grayscale with ROI: x={x}, y={y}, w={w}, h={h}")
            else:
                print(f"Video {video_name} not found, skipping.")
