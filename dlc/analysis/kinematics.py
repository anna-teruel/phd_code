"""
This script provides a comprehensive framework for calculating various kinematic variables such as speed, velocity, 
acceleration, displacement, and jerk from positional data over time. The script includes detailed functions organized 
within a Kinematics class, allowing for easy computation and retrieval of these kinematic metrics.
This framework is designed to be flexible and extendable, making it suitable for a wide range of applications in motion analysis.
@author Anna Teruel-Sanchis, 2024
"""

import numpy as np

class Kinematics:
    def __init__(self, df, bodypart):
        """
        Initialize the Kinematics object.

        Args:
            bodypart (np.ndarray): Array of positions (x, y) at different time points.
            times (np.ndarray): Array of time points corresponding to the positions.
        """
        self.df = df
        self.bodypart = bodypart
        self.scorer = df.columns.get_level_values(0)[0]  # Automatically detect the scorer
        self.positions = df[self.scorer][bodypart][['x', 'y']].values
        self.times = df.index.values

    def get_speed(self):
        """
        Calculate the speed at each time point.
        Spees is a sclarar quantity that represents how fast an object is moving. It is calculated as the distance traveled 
        per unit of time, without considering the direction of movement. Speed is always a positive value. 

        Returns:
            np.ndarray: Array of speeds at each time point.
        """
        distances = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        time_intervals = np.diff(self.times)
        speeds = distances / time_intervals
        return speeds

    def get_velocity(self):
        """
        Calculate the velocity at each time point.
        Velocity is a vector quantity that represents the rate of change of an object's position. It includes both speed and the 
        direction of movement. Velocity can take positive and negative values, depending on the direction of movement.

        Valocity can help in understanding the behavioral patterns of the animals. For example, sudden changes in velocity might 
        indicate a startle response or a change in behavior, such as escaping from a predator. 

        Returns:
            np.ndarray: Array of velocities at each time point.
        """
        displacements = np.diff(self.positions, axis=0)
        time_intervals = np.diff(self.times)[:, np.newaxis]
        velocities = displacements / time_intervals
        return velocities

    def get_acceleration(self):
        """
        Calculate the acceleration at each time point.
        Measures the rate of change of position, calculated as change in velocity divided by time. 

        Returns:
            np.ndarray: Array of accelerations at each time point.
        """
        velocities = self.calculate_velocity()
        velocity_changes = np.diff(velocities, axis=0)
        time_intervals = np.diff(self.times[1:])
        accelerations = velocity_changes / time_intervals[:, np.newaxis]
        return accelerations

    def get_displacement(self):
        """
        Calculate the displacement at each time point. Measures the rate of change of velocity, calculated as 
        the difference between consecutive positions, considering the direction of movement. 

        Displacement can be used to study the movement patterns of animals. For example, researchers can analyze 
        how far an animal moves from its starting point over a given period, which can provide insights into its 
        foraging behavior, territory range, or migration patterns.By measuring displacement, researchers can determine 
        how animals use their habitat. For instance, displacement data can reveal whether an animal tends to stay 
        within a small area or explores a larger territory, which can be important for understanding habitat preferences
        and requirements.

        Returns:
            np.ndarray: Array of displacements at each time point.
        """
        displacements = np.diff(self.positions, axis=0)
        return displacements

    def get_jerk(self):
        """
        Calculate the jerk at each time point.
        It is the derivative of acceleration, with respect to time. Then, it explains the rate at which an object's 
        acceleration changes with time. Jerk is a vector quantity, meaning it has both magnitude and direction. 
        
        In biomechanics, jerk is used to analyze the smoothnes of movement. High jerk values can indicate abrupt 
        changes in motion, which might be associated with discomfort or inefficiency in animal movement. Lower jerk
        values are often associated with smoother and more coordinated movements. 

        Returns:
            np.ndarray: Array of jerks at each time point.
        """
        accelerations = self.calculate_acceleration()
        acceleration_changes = np.diff(accelerations, axis=0)
        time_intervals = np.diff(self.times[2:])
        jerks = acceleration_changes / time_intervals[:, np.newaxis]
        return jerks
    
    def get_total_distance(self):
        """
        Calculate the total distance traveled by the object.
        Total distance is the cumulative distance traveled by summing up the distances between consecutive positions.

        Returns:
            float: Total distance traveled.
        """
        distances = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        return total_distance
