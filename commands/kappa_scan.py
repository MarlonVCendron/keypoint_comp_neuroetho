import keypoint_moseq as kpms # type: ignore
import numpy as np

from utils.print_legal import print_legal
from utils.load_data_and_config import load_data_and_config


def kappa_scan(
    project_dir,
    model_name,
    kappa_log_start,
    kappa_log_end,
    num_kappas,
    decrease_kappa_factor,
    num_ar_iters,
    num_full_iters,
    config_overrides=None,
):
    kappas = np.logspace(kappa_log_start, kappa_log_end, num_kappas)
    prefix = f"kappa_scan_{kappa_log_start}_{kappa_log_end}_{num_kappas}_{decrease_kappa_factor}"

    data, metadata, config = load_data_and_config(project_dir)
    if config_overrides:
        config.update(config_overrides)

    pca = kpms.load_pca(project_dir)

    print_legal(f"Iniciando scan de kappa com os valores: {kappas}")

    for kappa in kappas:
        print_legal(f"Ajustando modelo com kappa={kappa}")
        model_name = f"{prefix}-{kappa}"
        model = kpms.init_model(data, pca=pca, **config)

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
