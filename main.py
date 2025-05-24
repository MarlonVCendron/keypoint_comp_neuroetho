# Baseado em https://keypoint-moseq.readthedocs.io/en/latest/modeling.html

from utils.args import (
    project_dir,
    mixed_map_iters,
    model_name,
    num_ar_iters,
    command,
    parser,
    num_ar_iters_checkpoint,
    iters,
    kappa,
    base_iters,
    kappa_values,
)
from commands import init_project, fit_pca, fit_ar, fit_arhmm, kappa_scan, noise_calibration
from jax_moseq.utils import set_mixed_map_iters

set_mixed_map_iters(mixed_map_iters)


def main():
    config_overrides = {}

    if command == "init":
        init_project(project_dir)
    elif command == "noise_calibration":
        noise_calibration(project_dir)
    elif command == "fit_pca":
        fit_pca(project_dir, config_overrides=config_overrides)
    elif command == "fit_ar":
        fit_ar(project_dir, model_name, num_ar_iters, config_overrides=config_overrides)
    elif command == "fit_arhmm":
        fit_arhmm(
            project_dir,
            model_name,
            num_ar_iters_checkpoint,
            iters,
            kappa_val=kappa,
            config_overrides=config_overrides,
        )
    elif command == "kappa_scan":
        kappa_scan(
            project_dir,
            model_name,
            num_ar_iters_checkpoint,
            base_iters,
            kappa_values,
            config_overrides=config_overrides,
        )
    else:
        print(f"Comando desconhecido: {command}")
        parser.print_help()


if __name__ == "__main__":
    main()
