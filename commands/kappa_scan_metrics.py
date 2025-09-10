import keypoint_moseq as kpms  # type: ignore
from utils.load_data_and_config import load_data_and_config
import numpy as np

def kappa_scan_metrics(project_dir):
    model_names = [
        # "kappa_scan_3.0_7.0_5_10-1000.0",
        # "kappa_scan_3.0_7.0_5_10-10000.0",
        # "kappa_scan_3.0_7.0_5_10-100000.0",
        # "kappa_scan_3.0_7.0_5_10-1000000.0",
        # "kappa_scan_3.0_7.0_5_10-10000000.0",
        "kappa_scan_4.0_6.0_3_10-10000.0",
        "kappa_scan_5.0_6.0_2_10-100000.0",
    ]

    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
    best_model = model_names[np.argmax(eml_scores)]
    print(f"Best model: {best_model}")

    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)
