import keypoint_moseq as kpms
from utils import print_legal


def kappa_scan(
    project_dir_path,
    model_name_str,
    num_ar_iters_checkpoint,
    base_total_iters,
    kappa_values_str,
    config_overrides=None,
):
    print_legal("não implementado")
    # """Realiza uma scan de kappa: ajusta AR-HMM para múltiplos valores de kappa."""
    # print_legal(f"Realizando scan de kappa para o projeto: {project_dir_path}, nome do modelo: {model_name_str}")

    # try:
    #     kappa_values = [float(k.strip()) for k in kappa_values_str.split(',')]
    # except ValueError:
    #     print("Error: kappa_values must be a comma-separated list of numbers (e.g., '1e3,1e4,1e5').")
    #     return

    # if not kappa_values:
    #     print("Error: No kappa values provided for scanning.")
    #     return

    # for kappa_val in kappa_values:
    #     # Create a unique model name for each kappa value to avoid overwriting
    #     scan_model_name = f"{model_name_str}_kappa{kappa_val}"
    #     print(f"--- Starting fit for kappa={kappa_val}, model name: {scan_model_name} ---")
    #     kpms.fit_arhmm_model(
    #         project_dir_path,
    #         scan_model_name,
    #         num_ar_iters_checkpoint,
    #         base_total_iters, # Use the same total iterations for each scan
    #         kappa_val=kappa_val,
    #         config_overrides=config_overrides
    #     )
    #     print(f"--- Completed fit for kappa={kappa_val} ---")
    # print("Kappa scan finished.")


# kappas = np.logspace(3,7,5)
# decrease_kappa_factor = 10
# num_ar_iters = 50
# num_full_iters = 200

# prefix = 'my_kappa_scan'

# for kappa in kappas:
#     print(f"Fitting model with kappa={kappa}")
#     model_name = f'{prefix}-{kappa}'
#     model = kpms.init_model(data, pca=pca, **config())

#     # stage 1: fit the model with AR only
#     model = kpms.update_hypparams(model, kappa=kappa)
#     model = kpms.fit_model(
#         model,
#         data,
#         metadata,
#         project_dir,
#         model_name,
#         ar_only=True,
#         num_iters=num_ar_iters,
#         save_every_n_iters=25
#     )[0];

#     # stage 2: fit the full model
#     model = kpms.update_hypparams(model, kappa=kappa/decrease_kappa_factor)
#     kpms.fit_model(
#         model,
#         data,
#         metadata,
#         project_dir,
#         model_name,
#         ar_only=False,
#         start_iter=num_ar_iters,
#         num_iters=num_full_iters,
#         save_every_n_iters=25
#     );

# kpms.plot_kappa_scan(kappas, project_dir, prefix)
