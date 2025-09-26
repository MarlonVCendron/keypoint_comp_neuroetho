import keypoint_moseq as kpms  # type: ignore
import numpy as np
from scipy.ndimage import median_filter
from utils.load_data_and_config import load_data_and_config
from utils.find_medoid_distance_outliers import filter_outliers, plot_medoid_distance_outliers
from utils.print_legal import print_legal
from os import path


def outliers(project_dir):
    kpms.update_config(project_dir, outlier_scale_factor=6.0)

    data, metadata, config, coordinates, _, confidences = load_data_and_config(
        project_dir, remove_outliers=False
    )

    coordinates, confidences, outliers = filter_outliers(
        coordinates,
        confidences,
        config,
        lambda coordinates, confidences, outliers, recording_name, raw_coords: plot_medoid_distance_outliers(
            project_dir,
            recording_name,
            raw_coords,
            coordinates[recording_name],
            outliers["mask"],
            outliers["thresholds"],
            **config
        ),
    )
