"""

@author Anna Teruel-Sanchis, 2024
"""

import cv2
import numpy as np
import os
import glob
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt

class VideoPCA:
    def __init__(self, video_files):
        """
        Initialize the VideoPCA class with a list of video files.

        Args:
            video_files (list): List of paths to video files.

        Attributes:
            video_files (list): List of paths to video files.
            data (np.ndarray): Array of concatenated flattened frames from all videos.
            pca (PCA): Fitted PCA object.
            cumulative_variance (np.ndarray): Cumulative variance explained by the principal components.
        """
        self.video_files = video_files
        self.data = None
        self.pca = None
        self.cumulative_variance = None

    def flatten_frames(self, video_path):
        """
        Read a video file and return flattened grayscale frames.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: Array of flattened grayscale frames.

        Raises:
            ValueError: If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to grayscale and flatten it
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flat_frame = gray_frame.flatten()
            frames.append(flat_frame)

        cap.release()
        return np.array(frames)

    def collect_frames(self):
        """
        Collect and concatenate flattened frames from the list of video files.

        This method reads each video file, extracts and flattens the frames,
        and then concatenates all frames into a single dataset.

        Returns:
            None
        """
        all_frames = []
        for video_file in self.video_files:
            frames = self.flatten_frames(video_file)
            all_frames.append(frames)
        self.data = np.vstack(all_frames)

    def pca_videos(self):
        """
        Perform PCA on the collected data and calculate the cumulative variance explained.

        This method fits a PCA model to the data and computes the cumulative
        variance explained by the principal components.

        Returns:
            None

        Raises:
            ValueError: If data has not been collected.
        """
        if self.data is None:
            raise ValueError("Data not collected. Please run collect_frames() first.")

        self.pca = PCA()
        self.pca.fit(self.data)
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

    def plot_cumulative_variance(self):
        """
        Plot the cumulative variance explained by PCA components.

        This method creates a plot showing the cumulative variance explained
        by the principal components.

        Returns:
            None

        Raises:
            ValueError: If PCA has not been performed.
        """
        if self.cumulative_variance is None:
            raise ValueError("PCA not performed. Please run perform_pca() first.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Variance Explained by PCA Components')
        plt.grid(True)
        plt.show()


class VideoICA:
    def __init__(self, data):
        """
        Initialize the VideoICA class with the collected data.

        Args:
            data (np.ndarray): Concatenated array of flattened frames from all videos.
        
        Attributes:
            data (np.ndarray): Concatenated array of flattened frames from all videos.
            ica (FastICA): Fitted ICA object.
            sources (np.ndarray): Independent components obtained from ICA.
        """
        self.data = data
        self.ica = None
        self.sources = None

    def perform_ica(self, n_components):
        """
        Perform ICA on the collected data using the specified number of components.

        Args:
            n_components (int): Number of components to use for ICA.
        
        This method updates the `ica` attribute with the fitted ICA object and `sources` with the independent components.
        
        Raises:
            ValueError: If data has not been collected.
        """
        if self.data is None:
            raise ValueError("Data not collected. Please provide the data.")
        
        self.ica = FastICA(n_components=n_components, random_state=0)
        self.sources = self.ica.fit_transform(self.data)
        
    def plot_ica_components_as_images(self, output_dir='.', components_per_fig=100):
        """_summary_

        Args:
            output_dir (str, optional): _description_. Defaults to '.'.
            components_per_fig (int, optional): _description_. Defaults to 100.

        Raises:
            ValueError: _description_
        """        
        if self.sources is None:
            raise ValueError("ICA not performed. Please run perform_ica() first.")

        os.makedirs(output_dir, exist_ok=True)
        num_components = self.sources.shape[1]

        n_cols = 10
        n_rows = min(components_per_fig // n_cols, num_components)

        for fig_num in range(0, num_components, components_per_fig):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2))
            for i in range(components_per_fig):
                comp_idx = fig_num + i
                if comp_idx >= num_components:
                    break
                ax = axes[i // n_cols, i % n_cols]
                try:
                    component_image = self.ica.components_[comp_idx].reshape(self.frame_shape)
                except ValueError as e:
                    print(f"Error reshaping component {comp_idx}: {e}")
                    continue
                ax.imshow(component_image, cmap='jet', aspect='auto')
                ax.set_title(f'Component {comp_idx + 1}')
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ica_components_images_{fig_num // components_per_fig}.png'))
            plt.close()

    def identify_noise_components(self, num_components=5):
        """_summary_

        Args:
            num_components (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """        
        component_variances = np.var(self.sources, axis=0)
        print(f"Component variances: {component_variances}")
        noise_components = np.argsort(component_variances)[-num_components:]
        print(f"Top {num_components} noise components by variance: {noise_components}")
        return noise_components.tolist()

    def remove_noise_and_reconstruct(self, noise_components):
        """_summary_

        Args:
            noise_components (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """        
        if self.sources is None:
            raise ValueError("ICA not performed. Please run perform_ica() first.")
        cleaned_sources = self.sources.copy()
        cleaned_sources[:, noise_components] = 0
        reconstructed_data = self.ica.inverse_transform(cleaned_sources)
        return reconstructed_data

    def sum_non_discarded_and_reconstruct(self, noise_components):
        """_summary_

        Args:
            noise_components (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """        
        if self.sources is None:
            raise ValueError("ICA not performed. Please run perform_ica() first.")
        # Identify non-discarded components
        all_components = np.arange(self.sources.shape[1])
        non_discarded_components = np.setdiff1d(all_components, noise_components)
        print(f"Non-discarded components: {non_discarded_components}")

        # Sum non-discarded components to reconstruct the data
        reconstructed_data = self.ica.mixing_[:, non_discarded_components] @ self.sources[:, non_discarded_components].T
        reconstructed_data = reconstructed_data.T + self.ica.mean_

        return reconstructed_data

    def save_reconstructed_videos(self, cleaned_data, output_dir, suffix='_ica'):
        """_summary_

        Args:
            cleaned_data (_type_): _description_
            output_dir (_type_): _description_
            suffix (str, optional): _description_. Defaults to '_ica'.

        Raises:
            ValueError: _description_
        """        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate the number of frames per video
        frames_per_video = len(cleaned_data) // len(self.video_files)
        
        for idx, video_file in enumerate(self.video_files):
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file {video_file}")
            
            # Get the original frame rate
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Define the codec and create a VideoWriter object
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}{suffix}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, fps, (self.frame_shape[1], self.frame_shape[0]), isColor=False)
            
            for i in range(frames_per_video):
                frame_index = idx * frames_per_video + i
                frame_data = cleaned_data[frame_index].reshape(self.frame_shape)
                frame_data = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                out.write(frame_data)
            
            cap.release()
            out.release()
            print(f"Saved reconstructed video to {output_file}")

    def save_ica_components(self, output_dir):
        """_summary_

        Args:
            output_dir (_type_): _description_
        """        
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'ica_components.npy'), self.ica.components_)
        np.save(os.path.join(output_dir, 'ica_sources.npy'), self.sources)
        np.save(os.path.join(output_dir, 'ica_mixing.npy'), self.ica.mixing_)
        np.save(os.path.join(output_dir, 'ica_mean.npy'), self.ica.mean_)
        print(f"ICA components, sources, mixing matrix, and mean saved to {output_dir}")

    def load_ica_components(self, output_dir):
        """_summary_

        Args:
            output_dir (_type_): _description_
        """        
        if self.ica is None:
            self.ica = FastICA()  # Initialize the ICA object if not already done
        self.ica.components_ = np.load(os.path.join(output_dir, 'ica_components.npy'))
        self.sources = np.load(os.path.join(output_dir, 'ica_sources.npy'))
        self.ica.mixing_ = np.load(os.path.join(output_dir, 'ica_mixing.npy'))
        self.ica.mean_ = np.load(os.path.join(output_dir, 'ica_mean.npy'))
        print(f"ICA components, sources, mixing matrix, and mean loaded from {output_dir}")