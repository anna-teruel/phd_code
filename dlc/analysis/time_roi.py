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

    Attributes:
        video_path (str): Path to the video file.
        num_rois (int): Number of regions of interest to be drawn.
        roi_dataframe (pd.DataFrame or None): DataFrame containing the captured ROI data.
    """   
    def __init__(self, video_path, save_dir, num_rois=1):
        """
        Initializes the ROIDrawer object with the specified video path and number of ROIs.

        Args:
            video_path (str): The file path to the video from which a random frame is selected for drawing ROIs.
            num_rois (int, optional): The number of ROIs to be drawn. Defaults to 1.
        """         
        self.video_path = video_path
        self.num_rois = num_rois
        self.roi_dataframe = None
        self.save_dir = save_dir

    def on_button_click(self, shapes_layer):
        print("Button clicked, processing ROIs...")

        # Existing code to process ROIs and create DataFrame
        rois_data = shapes_layer.data
        roi_list = []
        for index, coords in enumerate(rois_data):
            for vertex_index, (x, y) in enumerate(coords):
                roi_list.append({'index': index, 'shape-type': 'polygon', 'vertex-index': vertex_index, 'axis-0': x, 'axis-1': y})
        self.roi_dataframe = pd.DataFrame(roi_list)

        # Define file paths for saving
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hdf5_filename = f"{os.path.splitext(os.path.basename(self.video_path))[0]}_{timestamp}.h5"
        self.h5_file_path = os.path.join(self.save_dir, hdf5_filename)
        frame_path = self.h5_file_path.replace('.h5', '.png')

        # Draw ROIs on the current frame
        for shape in rois_data:
            pts = np.array(shape, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.current_frame, [pts], True, (0, 255, 0), 2)  # Change color and thickness as needed

        # Save the ROI DataFrame and the frame with ROIs
        self.roi_dataframe.to_hdf(self.h5_file_path, key='coordinates', mode='w')
        cv2.imwrite(frame_path, self.current_frame)

        print("ROI DataFrame saved to:", self.h5_file_path)
        print("Frame with ROIs saved to:", frame_path)


    def draw_rois(self):
        """
        Launches the Napari viewer, allowing the user to draw ROIs on a randomly selected video frame.

        This method sets up the Napari viewer with an image layer displaying a random frame from the video.
        It allows the user to draw the specified number of ROIs, and captures them when the "Save ROIs" button is clicked.

        Returns:
            pandas.DataFrame: A DataFrame containing the captured ROI data with columns 'index', 'shape-type', 
                              'vertex-index', 'axis-0', 'axis-1', or None if no ROIs are drawn.
        """         
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(total_frames))
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        self.current_frame = frame

        viewer = napari.Viewer()
        viewer.add_image(frame)
        shapes_layer = viewer.add_shapes(name='ROIs')

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        btn = QPushButton("Save ROIs")
        layout.addWidget(btn)

        btn.clicked.connect(lambda: self.on_button_click(shapes_layer))
        btn.clicked.connect(lambda: viewer.close())

        viewer.window.add_dock_widget(widget)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hdf5_filename = f"{os.path.splitext(os.path.basename(self.video_path))[0]}_{timestamp}.h5"
        self.h5_file_path = os.path.join(self.save_dir, hdf5_filename)  # Update this line

        napari.run()

        frame_path = self.h5_file_path.replace('.h5', '.png')
        cv2.imwrite(frame_path, self.current_frame)

        return self.h5_file_path, frame_path 
    
    def batch_processing(self, directory, file_format='mp4'):
        video_files = [f for f in os.listdir(directory) if f.endswith(file_format)]
        for video_file in video_files:
            self.video_path = os.path.join(directory, video_file)
            print(f"Processing video: {video_file}")

            # Draw ROIs for each video
            hdf5_path, frame_path = self.draw_rois()
            print(f"Processed video {video_file}, ROIs saved to {hdf5_path}, frame saved to {frame_path}")



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
    def __init__(self):
        """
        Initializes a new TimeinRoi instance with an empty list of ROIs.
        """        
        self.rois = []

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
        return time_in_rois
    
    def time_in_rois_dir(self, directory, scorer, body_part):
        """
        Processes a directory of DeepLabCut HDF5 files to calculate the time spent in each ROI for a specified body part.
        This function iterates over all HDF5 files in the given directory that end with 'filtered.h5'. For each file, 
        it extracts the tracking data of the specified body part and calculates the time spent in each ROI defined in the 
        TimeInRoi object.

        Args:
            directory (str): The path to the directory containing the HDF5 files.
            scorer (str): The scorer name as per the HDF5 file's structure, used to identify the tracking data.
            body_part (str): The name of the body part to track (e.g., 'nose').

        Returns:
            pd.DataFrame: A DataFrame containing the results, with columns for file name, ROI index, and time spent in each ROI.

        Raises:
            Exception: If an error occurs while processing a file, the function prints the error message and continues with the next file.
        """       
        results = []
        for filename in os.listdir(directory):
            if filename.endswith('filtered.h5'):
                try:
                    file_path = os.path.join(directory, filename)
                    tracking_data = self.extract_tracking_data(file_path, scorer, body_part)
                    time_in_rois = self.time_in_rois(tracking_data)

                    for i, time_in_roi in enumerate(time_in_rois):
                        results.append({
                            'file': filename,
                            'roi_index': i,
                            'time_in_roi': time_in_roi
                        })
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
        return pd.DataFrame(results)