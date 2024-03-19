"""
This script provides a framework for defining regions of interest (ROIs) of various shapes on a video frame 
using the Napari library, determining if a point is within a given ROI, and calculating the time spent in 
each ROI. It also includes functionality for saving the ROI data and the annotated frame to a specified directory.
@author Anna Teruel-Sanchis, 2023
"""
import napari
import cv2
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from shapely.geometry import Point, Polygon, box
import pickle
import datetime

class ROIDrawer:
    """
    A class for drawing and capturing regions of interest (ROIs) on a video frame using Napari.

    This class provides functionality to draw ROIs on video frames and save the ROI data 
    and the annotated frame. It uses Napari for visualization and interaction.

    """   
    def __init__(self, video_path, save_dir, num_rois=1):
        """
        Initializes the ROIDrawer object with the specified video path, save directory, and number of ROIs.

        Args:
            video_path (str): The file path to the video for drawing ROIs.
            save_dir (str): The directory where ROI data and images will be saved.
            num_rois (int, optional): The number of ROIs to be drawn. Defaults to 1.
        """ 
        self.video_path = video_path
    
        self.save_dir = save_dir
        self.num_rois = num_rois
        self.current_frame = None
        self.viewer = None
        self.shapes_layer = None
        self.video_files = []
        self.directory = None 


    def draw_rois(self):
        """
        Sets up a Napari viewer for drawing ROIs on a video frame and saving the ROIs.

        Opens a random frame from the video file and displays it in a Napari viewer.
        Users can draw Regions of Interest (ROIs) on the frame. Once the ROIs are drawn,
        the 'Save ROIs' button saves the ROI data to an HDF5 file and the annotated frame to a PNG file.

        Args:
            None

        Returns:
            None
        """
        self.current_frame = None  # Reset the current frame before capturing another one
        print(f"Drawing ROIs for video: {self.video_path}")

        # Load the video frame
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(total_frames))
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        if not ret:
            print(f"Failed to capture frame from video: {self.video_path}")
            return

        self.current_frame = frame
        if self.viewer is None:
            self.viewer = napari.Viewer()
        else:
            self.viewer.layers.clear()  # Clear existing layers

        image_layer_name = f"Frame-{np.random.randint(10000)}"
        self.viewer.add_image(self.current_frame, name=image_layer_name)
        self.shapes_layer = self.viewer.add_shapes(name='ROIs')

        # Set up the widget with buttons
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        btn_save = QPushButton("Save ROIs")
        layout.addWidget(btn_save)
        self.viewer.window.add_dock_widget(widget)

        # Connect button to save function
        btn_save.clicked.connect(self.on_save_button_clicked)

    def on_save_button_clicked(self):
        """
        Processes and saves the drawn ROIs when the 'Save ROIs' button is clicked.

        This function processes the coordinates of the drawn ROIs, annotates them on the frame,
        and saves the ROI data to an HDF5 file. It also saves the annotated frame as a PNG image.
        Each ROI is labeled with its index number on the image.

        Args: 
            None
            
        Returns:
            None: Prints the paths to the saved HDF5 and PNG files.
        
        Raises:
            Exceptions related to file I/O operations or data format issues.
        """      
        rois_data = self.shapes_layer.data
        roi_type = self.shapes_layer.mode
        roi_list = []

        for index, roi in enumerate(rois_data):
            corrected_roi = [(y, x) for x, y in roi]
            for vertex_index, (x_corrected, y_corrected) in enumerate(corrected_roi):
                roi_list.append({
                    'index': index,
                    'shape-type': roi_type,
                    'vertex-index': vertex_index,
                    'axis-0': x_corrected,
                    'axis-1': y_corrected,
                })

        roi_dataframe = pd.DataFrame(roi_list)
        hdf5_filename = f"{os.path.splitext(os.path.basename(self.video_path))[0]}_roi.h5"
        h5_file_path = os.path.join(self.save_dir, hdf5_filename)
        roi_dataframe.to_hdf(h5_file_path, key='coordinates', mode='w')

        for roi_index, roi in enumerate(rois_data):
            corrected_roi = [(y, x) for x, y in roi]
            pts = np.array(corrected_roi, np.int32).reshape((-1, 1, 2))
            self.current_frame = cv2.polylines(self.current_frame, [pts], True, (0, 255, 0), 2)

            # Find a representative point to place the index label
            label_point = np.mean(pts, axis=0).ravel()
            cv2.putText(self.current_frame, f"roi{roi_index}", (int(label_point[0]), int(label_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frame_path = h5_file_path.replace('.h5', '.png')
        cv2.imwrite(frame_path, self.current_frame)

        # Close the viewer
        self.viewer.close()

        print(f"ROIs saved to {h5_file_path}")
        print(f"Frame saved to {frame_path}")

#Utility function
def extract_roi_data(df, shape_type, roi_callback):
    """
    Extracts regions of interest (ROIs) from a DataFrame based on a specific shape type.

    Args:
        df (pd.DataFrame): The DataFrame containing ROI data.
        shape_type (str): The type of shape to extract (e.g., 'add_ellipse').
        roi_callback (function): A callback function that takes a group from the DataFrame and returns a dictionary 
                                 with the ROI object and its parameters.

    Returns:
        tuple: A tuple containing two elements: a list of ROI objects and a dictionary of parameters for each ROI.
    """  
    rois = []
    parameters = {}
    for (index, stype), group in df:
        if stype == shape_type:
            param = roi_callback(group)
            rois.append(param['roi'])
            parameters[f"{shape_type}{index}"] = param['params']
    return rois, parameters

class ROI(ABC):
    """
    Abstract base class for a region of interest (ROI).
    """  
    @abstractmethod
    def is_point_inside_roi(self, point):
        """
        Determines if a given point is inside the ROI.

        Args:
            point (tuple): A tuple representing the (x, y) coordinates of the point to check.

        Returns:
            bool: True if the point is inside the ROI, False otherwise.
        """       
        pass

class EllipseROI(ROI):
    """
    Ellipse region of interest class
    """   
    def __init__(self, center, rad):
        """
        Initializing the ellipse roi

        Attributes:
            center (tuple): The (x, y) coordinates of the ellipse's center.
            rad (tuple): A tuple representing the semi-major and semi-minor radii of the ellipse.
        """         
        self.center = center
        self.rad = rad

    @classmethod
    def extract_ellipses(cls, df):
        """
        Class method to extract ellipses from a DataFrame (output from ROIDrawer)

        Args:
            df (pd.DataFrame): DataFrame containing ellipse data.

        Returns:
            tuple: A tuple containing a list of EllipseROI objects and a dictionary of their parameters.
        """       
        def ellipse_callback(group):
            axis_0_values = group['axis-0'].values
            axis_1_values = group['axis-1'].values
            center = ((axis_0_values.min() + axis_0_values.max()) / 2, 
                      (axis_1_values.min() + axis_1_values.max()) / 2)
            radii = ((axis_0_values.max() - axis_0_values.min()) / 2, 
                     (axis_1_values.max() - axis_1_values.min()) / 2)
            return {'roi': cls(center, radii), 'params': {'center': center, 'rad': radii}}
        return extract_roi_data(df, 'add_ellipse', ellipse_callback)

    def is_point_inside_roi(self, point):
        """
        Determines if a given point is inside the ellipse.

        Args:
            point (tuple): A tuple representing the (x, y) coordinates of the point to check.

        Returns:
            bool: True if the point is inside the ellipse, False otherwise.
        """        
        x, y = point
        h, k = self.center
        a, b = self.rad
        return ((x - h)**2 / a**2) + ((y - k)**2 / b**2) <= 1

class PolygonROI(ROI):
    """
    Represents a polygonal region of interest.
    """    
    def __init__(self, vertices):
        """
        Initializes a PolygonROI instance with the given vertices.

        The vertices should define the boundary of the polygon. This constructor creates a 
        shapely Polygon object that represents the polygonal region of interest.

        Args:
            vertices (list of tuples): A list where each tuple contains the (x, y) coordinates of a vertex 
                                       of the polygon. The vertices are expected to be in a sequence that 
                                       outlines the boundary of the polygon.
        """       
        self.polygon = Polygon(vertices)
    
    @classmethod
    def extract_polygons(cls, df):
        """
        Class method to extract polygons from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing polygon data.

        Returns:
            tuple: A tuple containing a list of PolygonROI objects and a dictionary of their parameters.
        """        
        def polygon_callback(group):
            print("Processing group:", group)  # Debug print
            vertices = group[['axis-0', 'axis-1']].values.tolist()
            return {'roi': cls(vertices), 'params': {'vertices': vertices}}

        polygons, parameters = extract_roi_data(df, 'add_polygon', polygon_callback)
        print("Extracted polygons:", polygons)  # Debug print
        return polygons, parameters


    def is_point_inside_roi(self, point):
        """
        Determines if a given point is inside the polygon.

        Args:
            point (tuple): A tuple representing the (x, y) coordinates of the point to check.

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """
        return self.polygon.contains(Point(point))

class RectangleROI(ROI):
    """
    Represents a rectangular region of interest
    """    
    def __init__(self, bottom_left, top_right):
        """
        Initializing the rectangular ROI.

        Attributes:
        bottom_left (tuple): The (x, y) coordinates of the bottom left corner of the rectangle.
        top_right (tuple): The (x, y) coordinates of the top right corner of the rectangle.
        """        
        self.bottom_left = bottom_left
        self.top_right = top_right

    @classmethod
    def extract_rectangles(cls, df):
        """
        Class method to extract rectangles from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing rectangle data.

        Returns:
            tuple: A tuple containing a list of RectangleROI objects and a dictionary of their parameters.
        """       
        def rectangle_callback(group):
            bottom_left = group.iloc[0][['axis-0', 'axis-1']].values.tolist()
            top_right = group.iloc[-1][['axis-0', 'axis-1']].values.tolist()
            return {'roi': cls(bottom_left, top_right), 'params': {'bottom_left': bottom_left, 'top_right': top_right}}
        return extract_roi_data(df, 'add_rectangle', rectangle_callback)

    def is_point_inside_roi(self, point):
        """
        Determines if a given point is inside the rectangle.

        Args:
            point (tuple): A tuple representing the (x, y) coordinates of the point to check.

        Returns:
            bool: True if the point is inside the rectangle, False otherwise.
        """       
        x, y = point
        x1, y1 = self.bottom_left
        x2, y2 = self.top_right
        return x1 <= x <= x2 and y1 <= y <= y2

class TimeinRoi:
    """
    A class to manage and compute time spent in various regions of interest (ROIs) 
    based on tracking data.
    """    
    def __init__(self, fps):
        """
        Initializes a new TimeinRoi instance with an empty list of ROIs.
        """        
        self.rois = []
        self.fps = fps

    def add_roi(self, roi):
        """
        Adds a region of interest (ROI) to the list of ROIs.

        Args:
            roi (ROI): An instance of an ROI class (EllipseROI, PolygonROI, RectangleROI).
        """        
        self.rois.append(roi)

    def extract_tracking_data(self, file_path, scorer, body_part):
        """
        Extracts tracking data for a specific body part from a DeepLabCut HDF5 file.

        Args:
            file_path (str): Path to the HDF5 file containing DeepLabCut tracking data.
            scorer (str): The scorer name as per the HDF5 file's structure.
            body_part (str): The name of the body part to track (e.g., 'nose').

        Returns:
            List of tuples: A list where each tuple contains the (x, y) coordinates.
        """       
        df = pd.read_hdf(file_path)
        x_coords = df[(scorer, body_part, 'x')]
        y_coords = df[(scorer, body_part, 'y')]
        return list(zip(x_coords, y_coords))
            
    def time_in_rois(self, tracking_data):
        """
        Calculates the time spent in each ROI based on the tracking data.

        Args:
            tracking_data (list of tuples): A list of (x, y) coordinates representing tracking data.

        Returns:
            List of int: A list where each element represents the time spent in the corresponding ROI.
        """        
        time_in_rois = [0] * len(self.rois)
        for point in tracking_data:
            for i, roi in enumerate(self.rois):
                if roi.is_point_inside_roi(point):
                    time_in_rois[i] += 1
                    
        time_seconds = [time / self.fps for time in time_in_rois]
        return time_seconds
    
    def time_in_rois_dir(self, directory, rois, scorer, body_part):
        """
        Processes a directory of DeepLabCut HDF5 files to calculate the time spent in each ROI for a specified body part.
        
        This function iterates over all HDF5 files in the given directory that end with 'filtered.h5'.
        For each file, it uses the provided ROIs to calculate the time spent in each region based on the tracking data.

        Args:
            directory (str): Path to the directory containing the HDF5 files.
            rois (dict): A dictionary mapping file base names to lists of ROI objects.
            scorer (str): The scorer name as per the HDF5 file's structure.
            body_part (str): The body part to track (e.g., 'nose').

        Returns:
            pd.DataFrame: A DataFrame with columns for file name, ROI index, and time spent in each ROI.
        
        Raises:
            Exception: If an error occurs while processing a file, the function prints the error message and continues with the next file.
        """      
        results = []
        for filename in os.listdir(directory):
            if filename.endswith('filtered.h5'):
                try:
                    base_name = filename.replace(scorer + '_filtered.h5', '')
                    if base_name in rois:
                        self.rois.clear()
                        for roi in rois[base_name]:
                            self.add_roi(roi)

                        # Process tracking data for this file
                        file_path = os.path.join(directory, filename)
                        tracking_data = self.extract_tracking_data(file_path, scorer, body_part)
                        time_in_rois = self.time_in_rois(tracking_data)

                        for i, time_in_roi in enumerate(time_in_rois):
                            results.append({
                                'file': filename,
                                'roi_index': i,
                                'time_in_roi': time_in_roi
                            })
                    else:
                        print(f"No corresponding ROI data for {filename}. Skipping...")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
        return pd.DataFrame(results)