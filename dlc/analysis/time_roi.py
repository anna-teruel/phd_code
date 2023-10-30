import napari

def draw_rois(image, num_rois):
    """
    Prompt the user to draw a specified number of regions of interest (roi) on an image using Napari.
    
    Args:
        image (numpy array): The image on which to draw the ROIs.
        num_rois (int): The number of ROIs the user should draw.
        
    Returns:
        list: List of drawn shapes as regions of interest.
    """

    viewer = napari.Viewer()
    viewer.add_image(image)
    shapes_layer = viewer.add_shapes(name='ROIs')

    print(f"Please draw {num_rois} regions of interest using Napari. Close the window when done.")

    napari.run()
    
    return shapes_layer.data


