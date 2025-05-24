import keypoint_moseq as kpms
from jax_moseq.utils import set_mixed_map_iters
from utils.args import project_dir, model_name, mixed_map_iters

set_mixed_map_iters(mixed_map_iters)

config = kpms.load_config(project_dir)

coordinates, confidences, bodyparts = kpms.load_keypoints(
    config['video_dir'],
    "deeplabcut",
    extension=".csv",
)

data, metadata = kpms.format_data(coordinates, confidences, **config)

# model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# results = kpms.extract_results(model, metadata, project_dir, model_name)

# results = kpms.apply_model(model, data, metadata, project_dir, model_name, **config)

results = kpms.load_results(project_dir, model_name)

# Trajectory plots
# kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config)

kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **config);

