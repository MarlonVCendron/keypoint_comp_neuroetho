import keypoint_moseq as kpms
from jax_moseq.utils import set_mixed_map_iters
set_mixed_map_iters(8)

project_dir = "project"
config = lambda: kpms.load_config(project_dir)

coordinates, confidences, bodyparts = kpms.load_keypoints(
    config()['video_dir'],
    "deeplabcut",
    extension=".csv",
)

data, metadata = kpms.format_data(coordinates, confidences, **config())

model_name = '2025_05_10-14_06_30'

# model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# results = kpms.extract_results(model, metadata, project_dir, model_name)

# results = kpms.apply_model(model, data, metadata, project_dir, model_name, **config())

results = kpms.load_results(project_dir, model_name)

# Trajectory plots
# kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())

kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **config());

