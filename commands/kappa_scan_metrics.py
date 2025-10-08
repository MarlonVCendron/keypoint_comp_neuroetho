import keypoint_moseq as kpms  # type: ignore
from utils.load_data_and_config import load_data_and_config
import numpy as np

def kappa_scan_metrics(project_dir):
    model_names = [
        '12k_no_tail',
        '9k'
    ]

    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
    best_model = model_names[np.argmax(eml_scores)]
    print(f"Best model: {best_model}")

    fig, ax = kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)
    fig.savefig(f"{project_dir}/eml_scores.png")

    results = [kpms.load_results(project_dir, model_name) for model_name in model_names]
    fig, ax = kpms.plot_confusion_matrix(*results)
    fig.savefig(f"{project_dir}/confusion_matrix.png")
