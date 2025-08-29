import keypoint_moseq as kpms  # type: ignore
import numpy as np

from utils.print_legal import print_legal
from utils.load_data_and_config import load_data_and_config


def kappa_scan_metrics(project_dir):
    model_names = [
        "kappa_scan_3_7_5_10-1000.0",
        "kappa_scan_3_7_5_10-10000.0",
        "kappa_scan_4.0_7_4_10-100000.0",
        "kappa_scan_4.0_7_4_10-1000000.0",
        "kappa_scan_4.0_7_4_10-10000000.0",
    ]

    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
    best_model = model_names[np.argmax(eml_scores)]
    print(f"Best model: {best_model}")

    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)
    return
