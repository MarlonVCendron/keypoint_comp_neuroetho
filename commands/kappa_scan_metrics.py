import keypoint_moseq as kpms  # type: ignore
from utils.load_data_and_config import load_data_and_config
import numpy as np

def kappa_scan_metrics(project_dir):
    model_names = [
        "kappa_scan_3.6_5.0_5_10-3981.0717055349733",
        "kappa_scan_3.6_5.0_5_10-8912.509381337459",
        "kappa_scan_3.6_5.0_5_10-19952.62314968879",
        "kappa_scan_3.6_5.0_5_10-44668.359215096345",
        "kappa_scan_3.6_5.0_5_10-100000.0"
    ]

    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
    best_model = model_names[np.argmax(eml_scores)]
    print(f"Best model: {best_model}")

    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)
