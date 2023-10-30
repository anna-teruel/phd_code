import napari
import cv2
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

class ROIDrawer:
    """
    A class for drawing and capturing regions of interest (ROIs) on a video frame using Napari.

    Attributes:
        video_path (str): Path to the video file.
        num_rois (int): Number of regions of interest to be drawn.
        roi_dataframe (pd.DataFrame or None): DataFrame containing the captured ROI data.
    """   
    def __init__(self, video_path, num_rois=1):
        """
        Initializes the ROIDrawer object with the specified video path and number of ROIs.

        Args:
            video_path (str): The file path to the video from which a random frame is selected for drawing ROIs.
            num_rois (int, optional): The number of ROIs to be drawn. Defaults to 1.
        """         
        self.video_path = video_path
        self.num_rois = num_rois
        self.roi_dataframe = None

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
        return self.roi_dataframe

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

        # Create a widget for the button
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        btn = QPushButton("Save ROIs")
        layout.addWidget(btn)

        # Connect button click to a lambda function
        btn.clicked.connect(lambda: self.on_button_click(shapes_layer))
        btn.clicked.connect(lambda: viewer.close())

        viewer.window.add_dock_widget(widget)

        napari.run()
        return self.roi_dataframe
