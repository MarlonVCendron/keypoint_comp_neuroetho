# Baseado em https://keypoint-moseq.readthedocs.io/en/latest/modeling.html
from utils.args import build_parser, parser, get_arg
from commands import init_project, fit_pca, fit_arhmm, fit_full_model, kappa_scan, kappa_scan_metrics, results, outliers
from jax_moseq.utils import set_mixed_map_iters
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def main():
    build_parser()

    config_overrides = {}

    command = get_arg('command')
    project_dir = get_arg('project_dir')
    model_name = get_arg('model_name')
    num_ar_iters = get_arg('num_ar_iters')
    checkpoint = get_arg('checkpoint')
    iters = get_arg('iters')
    kappa = get_arg('kappa')
    kappa_log_start = get_arg('kappa_log_start')
    kappa_log_end = get_arg('kappa_log_end')
    num_kappas = get_arg('num_kappas')
    decrease_kappa_factor = get_arg('decrease_kappa_factor')
    mixed_map_iters = get_arg('mixed_map_iters')

    set_mixed_map_iters(mixed_map_iters)

    if command == "init":
        init_project(project_dir=project_dir)
    elif command == "outliers":
        outliers(project_dir=project_dir)
    elif command == "fit_pca":
        fit_pca(project_dir=project_dir, config_overrides=config_overrides)
    elif command == "kappa_scan":
        kappa_scan(
            project_dir=project_dir,
            model_name=model_name,
            kappa_log_start=kappa_log_start,
            kappa_log_end=kappa_log_end,
            num_kappas=num_kappas,
            decrease_kappa_factor=decrease_kappa_factor,
            num_ar_iters=num_ar_iters,
            num_iters=iters,
            config_overrides=config_overrides,
        )
    elif command == "kappa_scan_metrics":
        kappa_scan_metrics(project_dir=project_dir)
    elif command == "fit_arhmm":
        fit_arhmm(
            project_dir=project_dir,
            model_name=model_name,
            iters=iters,
            kappa=kappa,
            config_overrides=config_overrides,
        )
    elif command == "fit_full_model":
        fit_full_model(
            project_dir=project_dir,
            model_name=model_name,
            checkpoint=checkpoint,
            iters=iters,
            kappa=kappa,
            config_overrides=config_overrides,
        )
    elif command == "results":
        results(
            project_dir=project_dir,
            model_name=model_name,
            checkpoint=checkpoint,
            config_overrides=config_overrides,
        )
    else:
        print(f"Comando desconhecido: {command}")
        parser.print_help()


if __name__ == "__main__":
    main()
