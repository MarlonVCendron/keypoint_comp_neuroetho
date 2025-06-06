import keypoint_moseq as kpms # type: ignore # type: ignore
import numpy as np

from utils.print_legal import print_legal
from utils.load_data_and_config import load_data_and_config


def kappa_scan(
    project_dir,
    model_name,
    kappa_values_str,
    config_overrides=None,
):
    kappas = np.logspace(3, 7, 5)
    decrease_kappa_factor = 10
    num_ar_iters = 50
    num_full_iters = 200
    prefix = "kappa_scan"

    data, metadata, config = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    pca = kpms.load_pca(project_dir)

    for kappa in kappas:
        print_legal(f"Ajustando modelo com kappa={kappa}")
        model_name = f"{prefix}-{kappa}"
        model = kpms.init_model(data, pca=pca, **config())

        # Ajusta o modelo AR
        model = kpms.update_hypparams(model, kappa=kappa)
        model = kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            ar_only=True,
            num_iters=num_ar_iters,
            save_every_n_iters=25,
        )[0]

        # Ajusta o modelo AR-HMM
        model = kpms.update_hypparams(model, kappa=kappa / decrease_kappa_factor)
        kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            ar_only=False,
            start_iter=num_ar_iters,
            num_iters=num_full_iters,
            save_every_n_iters=25,
        )

    kpms.plot_kappa_scan(kappas, project_dir, prefix)
