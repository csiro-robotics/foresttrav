# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

import numpy as np
import copy


class DataPatchAugmenter:
    """
    A class to augment data patches. It will generate a deep copy of  data class without modifying the input data.
    Two different sets of data augmentation
        i) spatially which is applied to a whole patch, normally just voxel coordinates
        ii) Features which is applied to all a features of an associated coordinate

    """

    def __init__(self,
                 voxel_size:float,
                 n_voxel_displacement: int,
                 sample_pruning_chance: float,
                 translation_chance : float,
                 rotation_chance: float,
                 mirror_chance: float,
                 noise_chance: float, 
                 noise_mean: float,
                 noise_std: float,
                 
                 
                 ) -> None:
        self._sample_pruning_chance = sample_pruning_chance
        self._translation_chance = translation_chance
        self._rotation_chance = rotation_chance
        self._mirror_chance = mirror_chance
        self._noise_chance = noise_chance
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._voxel_size = voxel_size
        
        self.translation_options = np.array(
            [
                float(i)
                for i in range(
                    -1 * int(n_voxel_displacement),
                    int(n_voxel_displacement),
                )
                if i != 0
            ],
            dtype=np.float64,
        )
    

    def augment_data(self, coord: np.array, labels: np.array, features: np.array):
        """Data set is pruned and then augmented.

        Returns:
            coord_augmented:    augmented coordinates,
            labels:             labels are currently not augmented but passed on for completeness
            feature_arr:        feature data augmented
        """

        # Generates a prunning mask for all the points
        pruning_mask = (
            np.random.uniform(low=0.0, high=1.0, size=(coord.shape[0]))
            > self._sample_pruning_chance
        )

        # Store the pruning mask if needed by others (bad but necessary)
        self.pruning_mask = pruning_mask
        
        return (
            self.augment_coordinates(coord[pruning_mask]),
            labels[pruning_mask],
            self.add_random_noise_to_features(features[pruning_mask]),
        )

    def augment_coordinates(self, coord: np.array):
        """Generates a tranformation matrix to augment all coord of points in a batch"""

        # Add random Rotation
        T = np.eye(4)
        if (
            np.random.uniform(low=0.0, high=1.0, size=(1, 1))
            < self._rotation_chance
        ):
            T = np.matmul(self.rotation_matrix_ez(), T)

        # TODO: Add mirror ex/ey
        T = np.matmul(self.mirror(), T)

        if (
            np.random.uniform(low=0.0, high=1.0, size=(1, 1))
            < self._translation_chance
        ):
            T = np.matmul(self.random_translation(), T)

        return np.matmul(coord, T[0:3, 0:3]) + T[0:3, 3:4].transpose()

    def random_translation(self):
        "Random translation between trans_low and trans_high in x,y axis"
        t = np.zeros(shape=(3, 1))
        t[0] = np.random.choice(self.translation_options)
        t[1] = np.random.choice(self.translation_options)
        T = np.eye(4)
        T[0:3, 3:4] = t * self._voxel_size
        return T

    def rotation_matrix_ez(self):
        "Random rotation between alpha_low and alpha high around ez for 90, 180, 270 degrees"
        n = np.random.randint(low=0, high=3)
        alpha = np.pi / 2.0 * float(n)
        return np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0, 0],
                [np.sin(alpha), np.cos(alpha), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def mirror(self):
        # Mirro around ex and ey, we avoid ez since we assume gravity alignes of the data
        T = np.eye(4)
        if np.random.uniform(low=0.0, high=1.0, size=(1, 1)) < self._mirror_chance:
            T[0][0] = -1.0
        if np.random.uniform(low=0.0, high=1.0, size=(1, 1)) < self._mirror_chance:
            T[1][1] = -1.0
        return T

    def add_random_noise_to_features(self, data_set: np.array):
        """Adds random gaussian noise to features with a certain mean, variance"""

        if self._noise_chance <= 0.0:
            return data_set

        # Abort early to avoid this copy
        data_set_mod = copy.deepcopy(data_set)

        mask = (
            np.random.uniform(low=0.0, high=1.0, size=(data_set.shape[0]))
            < self._noise_chance
        )

        noise = np.random.normal(
            self._noise_mean,
            self._noise_std,
            size=(np.count_nonzero(mask), data_set.shape[1]),
        )
        # Ensure that data is 0-1
        data_set_mod[mask] = np.clip(data_set_mod[mask] + noise, 0, 1)

        return data_set_mod
