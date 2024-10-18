"""
Raw video preprocessing functions for DeepLabCut data.This module provides a set of raw video preprocessing functions 
tailored for cropping and scaling videos. The functions facilitate the extraction of frames from videos, manual selection 
of Regions of Interest (ROIs) from images, and subsequent video processing based on these ROIs. The primary goal is to 
streamline the preprocessing steps required for various video analysis tasks, making it easier to crop, scale, and convert
 videos to grayscale based on user-defined ROIs.
@author Anna Teruel-Sanchis, october 2024
"""
import os
import csv
import subprocess
import cv2

#TODO write a demo notebook for this class @Anna

class VideoPreprocessor:
    """
    A class for preprocessing raw videos for DeepLabCut data.

    This class provides methods to extract frames from videos, manually select Regions of Interest (ROIs) from images,
    and process videos based on these ROIs. The primary goal is to streamline the preprocessing steps required for
    DeepLabCut analysis, making it easier to crop, scale, and convert videos to grayscale based on user-defined ROIs.

    Attributes:
        video_folder (str): The path to the folder containing the videos.
        image_folder (str): The path to the folder containing the images.
        output_csv (str): The path to the CSV file where the selected ROI coordinates will be saved.
        roi_csv (str): The path to the CSV file containing the ROI coordinates for each image.
    """

    def __init__(self, video_folder, image_folder=None, output_csv=None, roi_csv=None):
        self.video_folder = video_folder
        self.image_folder = image_folder
        self.output_csv = output_csv
        self.roi_csv = roi_csv

    def extract_frame_from_videos(self, output_format='jpeg', frame_time='00:00:05'):
        """
        Extract a frame from each video in the specified folder and save it as an image.

        This method processes each video in the given folder, extracting a frame from the
        specified time (default is the 5th second). The extracted frame is saved as an image
        in the specified format (default is 'jpeg').

        Args:
            output_format (str, optional): The format in which to save the extracted frame. Defaults to 'jpeg'.
            frame_time (str, optional): The time in the video from which to extract the frame (format: 'HH:MM:SS'). 
                Defaults to '00:00:05'.
        """
        for file_name in os.listdir(self.video_folder):
            if file_name.endswith('.mp4'):
                video_path = os.path.join(self.video_folder, file_name)
                output_image_path = os.path.join(self.video_folder, f"{file_name[:-4]}.{output_format}")

                ffmpeg_command = [
                    'ffmpeg', '-y',  
                    '-i', video_path,  
                    '-ss', frame_time, 
                    '-vframes', '1',  
                    output_image_path  
                ]

                subprocess.run(ffmpeg_command)
                print(f"Extracted frame from {file_name} and saved it as {output_image_path}")

    def draw_rois(self):
        """
        Manually select Regions of Interest (ROIs) from a set of images and save them to a CSV file.

        This method processes all images in the given folder and allows the user to interactively
        select ROIs for each image. The ROI coordinates (x, y, width, height) are stored in a CSV file
        in the given directory. This method can only read "jpeg" images. Other formats will make the method fail.
        """
        if not self.image_folder or not self.output_csv:
            raise ValueError("Both image_folder and output_csv must be specified.")

        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["image", "x", "y", "w", "h"])  

            for image_name in os.listdir(self.image_folder):
                if image_name.endswith('.jpeg'):  
                    image_path = os.path.join(self.image_folder, image_name)
                    image = cv2.imread(image_path)

                    if image is None:
                        print(f"Failed to load image: {image_name}")
                        continue

                    roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
                    cv2.destroyAllWindows()  # Close OpenCV windows
                    
                    x, y, w, h = roi

                    writer.writerow([image_name, x, y, w, h])
                    print(f"Saved ROI for {image_name}: x={x}, y={y}, w={w}, h={h}")

    def process_videos_based_on_roi(self):
        """
        Crop, scale, and convert videos to grayscale based on the Regions of Interest (ROIs) stored in a CSV file.

        This method processes videos in the specified folder based on the ROI coordinates provided in the CSV file.
        It applies cropping, scaling, and grayscale conversion to each video. Processed videos are saved with a modified name.

        This method will only process "mp4" files. Other formats will raise an error.
        """
        if not self.roi_csv:
            raise ValueError("roi_csv must be specified.")

        with open(self.roi_csv, mode='r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                image_name = row['image']
                x = int(row['x'])
                y = int(row['y'])
                w = int(row['w'])
                h = int(row['h'])

                video_name = image_name.replace(".jpeg", ".mp4").replace(".jpg", ".mp4").replace(".png", ".mp4")
                video_path = os.path.join(self.video_folder, video_name)
                output_video_path = os.path.join(self.video_folder, video_name.replace(".mp4", "_cropped_scaled_gray.mp4"))

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