# Baseado em https://keypoint-moseq.readthedocs.io/en/latest/modeling.html

import keypoint_moseq as kpms # type: ignore
from jax_moseq.utils import set_mixed_map_iters
from utils.args import project_dir, mixed_map_iters, fit_ar_only, model_name, num_ar_iters

set_mixed_map_iters(mixed_map_iters)

config = kpms.load_config(project_dir)

coordinates, confidences, bodyparts = kpms.load_keypoints(
    config['video_dir'],
    "deeplabcut",
    extension=".csv",
)

data, metadata = kpms.format_data(coordinates, confidences, **config)

pca = kpms.load_pca(project_dir)

model = kpms.init_model(data, pca=pca, **config)

if fit_ar_only:
  model, model_name = kpms.fit_model(
      model,
      data,
      metadata,
      project_dir,
      model_name,
      ar_only=True,
      num_iters=num_ar_iters,
  )
  exit()
else:
  # Carrega o modelo do checkpoint, começa a partir das iterações do modelo AR
  model, data, metadata, current_iter = kpms.load_checkpoint(
      project_dir,
      model_name,
      iteration=num_ar_iters
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
