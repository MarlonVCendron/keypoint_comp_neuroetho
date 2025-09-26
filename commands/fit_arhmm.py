import keypoint_moseq as kpms  # type: ignore
import numpy as np
from scipy.ndimage import median_filter
from utils.load_data_and_config import load_data_and_config
from utils.print_legal import print_legal
from os import path


def fit_arhmm(project_dir, model_name, iters, kappa=None, config_overrides=None):
    """Ajusta o modelo AR-HMM inicial."""

    print_legal(
        f"Ajustando o modelo AR-HMM para o projeto: {project_dir}, nome do modelo: {model_name}, iterações: {iters}"
    )

    data, metadata, config, coordinates, video_frame_indexes, _ = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    # Estima o sigmasq_loc
    sigmasq_loc = estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config["fps"])
    kpms.update_config(project_dir, sigmasq_loc=sigmasq_loc)

    pca = kpms.load_pca(project_dir)

    model = kpms.init_model(data, pca=pca, **config)

    kappa_to_use = kappa if kappa is not None else config.get("kappa", 1e4)
    model = kpms.update_hypparams(model, kappa=kappa_to_use)
    print_legal(f"Usando kappa: {kappa_to_use}")

    print_legal(f"Iniciando ajuste do modelo AR-HMM com {iters} iterações.")

    model, _ = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=True,
        num_iters=iters,
        parallel_message_passing=False,  # SALVA USO DE MEMÓRIA https://keypoint-moseq.readthedocs.io/en/latest/FAQs.html#out-of-memory
    )

    print_legal(f"Ajuste do modelo AR-HMM completo. Modelo salvo como {model_name}.")

# De https://keypoint-moseq.readthedocs.io/en/latest/_modules/keypoint_moseq/util.html#estimate_sigmasq_loc
def estimate_sigmasq_loc(Y, mask, filter_size=30) -> float:
    masked_centroids = np.where(mask[:, :, None], np.median(Y, axis=2), np.nan)
    smoothed_centroids = median_filter(masked_centroids, (1, filter_size, 1))
    distances = np.linalg.norm(np.diff(smoothed_centroids, axis=1), axis=-1)  # (batch, frames)
    return float(np.nanmean(distances)**2)