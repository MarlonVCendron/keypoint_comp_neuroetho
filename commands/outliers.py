import keypoint_moseq as kpms  # type: ignore
import numpy as np
from scipy.ndimage import median_filter
from utils.load_data_and_config import load_data_and_config
from utils.find_medoid_distance_outliers import filter_outliers, plot_medoid_distance_outliers, plot_keypoint_distance_outliers
from utils.print_legal import print_legal
from os import path


def outliers(project_dir):
    data, metadata, config, coordinates, _, confidences = load_data_and_config(
        project_dir, remove_outliers=False
    )

    coordinates, confidences, outliers = filter_outliers(
        coordinates,
        confidences,
        config,
        lambda coordinates, confidences, combined_outliers, recording_name, raw_coords: plot_outliers(
            project_dir, config, coordinates, confidences, combined_outliers, recording_name, raw_coords
        ),
        project_dir=project_dir,
    )


def plot_outliers(
    project_dir, config, coordinates, confidences, combined_outliers, recording_name, raw_coords
):
    # plot_medoid_distance_outliers(
    #     project_dir,
    #     recording_name,
    #     raw_coords,
    #     coordinates[recording_name],
    #     combined_outliers["medoid_outliers"]["mask"],
    #     combined_outliers["medoid_thresholds"],
    #     **config
    # )
    print_keypoint_distance_outlier_summary(recording_name, combined_outliers)


def print_keypoint_distance_outlier_summary(recording_name, combined_outliers):
    medoid_mask = combined_outliers["medoid_outliers"]["mask"]
    keypoint_distance_mask = combined_outliers["keypoint_distance_outliers"]["mask"]
    combined_mask = combined_outliers["mask"]
    
    n_frames, n_keypoints = medoid_mask.shape
    total_keypoints = n_frames * n_keypoints
    
    medoid_outliers = np.sum(medoid_mask)
    keypoint_distance_outliers = np.sum(keypoint_distance_mask)
    combined_outliers_count = np.sum(combined_mask)
    
    medoid_percentage = (medoid_outliers / total_keypoints) * 100
    keypoint_distance_percentage = (keypoint_distance_outliers / total_keypoints) * 100
    combined_percentage = (combined_outliers_count / total_keypoints) * 100
    
    print(f"\n=== Outliers para {recording_name} ===")
    print(f"Keypoints: {total_keypoints:,} ({n_frames:,} frames x {n_keypoints} keypoints)")
    print(f"Medoid distance outliers: {medoid_outliers:,} ({medoid_percentage:.2f}%)")
    print(f"Keypoint distance outliers: {keypoint_distance_outliers:,} ({keypoint_distance_percentage:.2f}%)")
    print(f"Combined outliers: {combined_outliers_count:,} ({combined_percentage:.2f}%)")
    
    for i in range(n_keypoints):
        keypoint_medoid = np.sum(medoid_mask[:, i])
        keypoint_distance = np.sum(keypoint_distance_mask[:, i])
        keypoint_combined = np.sum(combined_mask[:, i])
        
        medoid_pct = (keypoint_medoid / n_frames) * 100
        distance_pct = (keypoint_distance / n_frames) * 100
        combined_pct = (keypoint_combined / n_frames) * 100
        
        print(f"  Keypoint {i}: Medoid={keypoint_medoid:,} ({medoid_pct:.1f}%), "
              f"Distance={keypoint_distance:,} ({distance_pct:.1f}%), "
              f"Combined={keypoint_combined:,} ({combined_pct:.1f}%)")
    
    print("=" * 60)