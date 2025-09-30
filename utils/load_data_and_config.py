import keypoint_moseq as kpms # type: ignore
from utils.print_legal import print_legal
from utils.video_frame_indexes import get_video_frame_indexes
from utils.find_medoid_distance_outliers import filter_outliers

def load_config(project_dir):
    """Carrega a configuração do projeto."""
    return kpms.load_config(project_dir)

def load_keypoints(project_dir):
    """Carrega os keypoints do DLC."""
    config = load_config(project_dir)
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        config["video_dir"], "deeplabcut", extension=".csv"
    )
    return coordinates, confidences, bodyparts

def load_data_and_config(project_dir, remove_outliers=True):
    """Carrega a configuração e os keypoints do DLC."""
    print_legal(f"Carregando configuração e keypoints do DLC de: {project_dir}")

    kpms.update_config(project_dir, outlier_scale_factor=12.0)

    config = load_config(project_dir)
    coordinates, confidences, bodyparts = load_keypoints(project_dir)
    coordinates, confidences, video_frame_indexes = get_video_frame_indexes(coordinates, confidences)
    if remove_outliers:
        coordinates, confidences, _ = filter_outliers(coordinates, confidences, config)
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    return data, metadata, config, coordinates, video_frame_indexes, confidences
