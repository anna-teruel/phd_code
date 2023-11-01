import napari
import cv2
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from shapely.geometry import Point, Polygon, box

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
        """
        Handles the button click event to capture and process the drawn ROIs.

        This method is triggered when the "Save ROIs" button is clicked. It processes the ROIs drawn on the viewer,
        converts them into a DataFrame, and assigns it to self.roi_dataframe.

        Args:
            shapes_layer (napari.layers.Shapes): The Napari layer containing the drawn ROIs.
        """       
        print("Button clicked, processing ROIs...")  # Indicate that the method is being called

        rois_data = shapes_layer.data
        rois_types = [shapes_layer.mode] * len(rois_data)

        roi_list = []
        for index, (roi_type, coords) in enumerate(zip(rois_types, rois_data)):
            for vertex_index, (x, y) in enumerate(coords):
                roi_list.append({'index': index, 'shape-type': roi_type, 'vertex-index': vertex_index, 'axis-0': x, 'axis-1': y})

        self.roi_dataframe = pd.DataFrame(roi_list)  # Save the processed data directly into self.roi_dataframe

        print("ROI DataFrame assigned:", self.roi_dataframe)
        
        h5_file_path = self.save_dir
        self.roi_dataframe.to_hdf(h5_file_path, key='coordinates', mode='w')
        print("ROI DataFrame saved to:", h5_file_path)

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

        napari.run()
        return self.roi_dataframe


class TimeInROI:
    def __init__(self):
        """
        Initialize a TimeInROI object to manage regions of interest (ROIs).
        """
        self.rois = []
        self.roi_shapes = []

    def add_roi(self, shape, coordinates):
        """
        Add a region of interest.

        Args:
            shape (str): Either 'ellipse', 'polygon', or 'rectangle'.
            coordinates (tuple/list): A tuple or list of coordinates defining the ROI.
        """ 
        self.rois.append(coordinates)
        self.roi_shapes.append(shape)

    def is_point_inside_roi(self, point, roi_index):
        """
        Check if a point is inside a specified ROI.

        Args:
            point (tuple/list): A tuple or list representing the (x, y) coordinates of the point to check.
            roi_index (int): The index of the ROI to check against.

        Raises:
            ValueError: If the ROI shape is unknown or unsupported.

        Returns:
            bool: True if the point is inside the ROI. False otherwise.
        """        
        shape = self.roi_shapes[roi_index]
        coordinates = self.rois[roi_index]

        if shape == 'ellipse':
            return self._is_point_in_ellipse(point, *coordinates)
        elif shape == 'polygon':
            return self._is_point_in_polygon(point, coordinates)
        elif shape == 'rectangle':
            return self._is_point_in_rectangle(point, coordinates)
        else:
            raise ValueError(f"Unknown shape: {shape}")

    @staticmethod
    def _is_point_in_ellipse(p, center, radii):
        """
        Check if a point is inside an ellipse.

        Args:
            p (tuple/list): A tuple or list representing the (x, y) coordinates of the point to check.
            center (tuple/list): A tuple or list representing the (x, y) coordinates of the ellipse's center.
            radii (tuple/list): A tuple or list representing the semi-major and semi-minor radii of the ellipse.

        Returns:
            bool: True if the point is inside the ellipse, False otherwise.
        """        
        x, y = p
        h, k = center
        a, b = radii
        return ((x - h)**2 / a**2) + ((y - k)**2 / b**2) <= 1

    @staticmethod
    def _is_point_in_polygon(p, polygon):
        """
        Check if a point is inside a polygon using Shapely.

        Args:
            p (tuple/list): A tuple or list representing the (x, y) coordinates of the point to check.
            polygon (list): A list of tuples, each representing a vertex of the polygon.

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """ 
        point = Point(p)
        poly = Polygon(polygon)
        
        return poly.contains(point)

    @staticmethod
    def _is_point_in_rectangle(p, rectangle):
        """
        Check if a point is inside a rectangle.
        A rectangle is a polygon, so we could use previously defined function for a polygon.

        Args:
            p (tuple/list): A tuple or list representing the (x, y) coordinates of the point to check.
            rectangle (list): A list containing two tuples, each representing a corner of the rectangle.

        Returns:
            bool: True if the point is inside the rectangle, False otherwise.
        """       
        # Assuming the rectangle is represented as [(x1, y1), (x2, y2)]
        x, y = p
        (x1, y1), (x2, y2) = rectangle
        return x1 <= x <= x2 and y1 <= y <= y2

