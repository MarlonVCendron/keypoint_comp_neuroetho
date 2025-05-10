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

pca = kpms.load_pca(project_dir)

model = kpms.init_model(data, pca=pca, **config())

fit_ar_only = False
num_ar_iters = 50

if fit_ar_only:
    model, model_name = kpms.fit_model(
        model, data, metadata, project_dir, ar_only=True, num_iters=num_ar_iters
    )
else:
    model_name = '2025_05_10-14_06_30'
    # load model checkpoint
    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir, model_name, iteration=num_ar_iters
    )

# modify kappa to maintain the desired syllable time-scale
model = kpms.update_hypparams(model, kappa=1e4)

# run fitting for an additional 500 iters
model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + 500,
)[0]