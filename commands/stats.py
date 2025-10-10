import keypoint_moseq as kpms  # type: ignore
import os
import pandas as pd
import numpy as np

from utils.load_data_and_config import load_config

min_frequency = 0.005


def stats(project_dir, model_name):
    config = load_config(project_dir)
    save_dir = os.path.join(project_dir, model_name)

    # index.csv para rotular os grupos
    kpms.interactive_group_setting(project_dir, model_name)

    moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
    sessions = get_session_data(project_dir, model_name)
    moseq_df["session"] = np.concatenate(sessions)
    moseq_df.to_csv(os.path.join(save_dir, "moseq_df.csv"), index=False)
    print("Saved `moseq_df` dataframe to", save_dir)

    stats_df = kpms.compute_stats_df(
        project_dir,
        model_name,
        moseq_df,
        min_frequency=min_frequency,
        groupby=["group", "name"],  # column(s) to group the dataframe by
        fps=config["fps"],
    )
    stats_df.to_csv(os.path.join(save_dir, "stats_df.csv"), index=False)
    print("Saved `stats_df` dataframe to", save_dir)

    # syll_info.csv
    # kpms.label_syllables(project_dir, model_name, moseq_df)

    stats = ["frequency", "duration", "heading_mean", "angular_velocity_mean", "velocity_px_s_mean"]
    for stat in stats:
        kpms.plot_syll_stats_with_sem(
            stats_df,
            project_dir,
            model_name,
            plot_sig=True,  # whether to mark statistical significance with a star
            thresh=0.05,  # significance threshold
            stat=stat,  # statistic to be plotted (e.g. 'duration' or 'velocity_px_s_mean')
            order="stat",  # order syllables by overall frequency ("stat") or degree of difference ("diff")
            ctrl_group="s",  # name of the control group for statistical testing
            exp_group="t",  # name of the experimental group for statistical testing
            figsize=(8, 4),  # figure size
            groups=stats_df["group"].unique(),  # groups to be plotted
        )

    normalize = "bigram"  # normalization method ("bigram", "rows" or "columns")

    trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
        project_dir,
        model_name,
        normalize=normalize,
        min_frequency=min_frequency,  # minimum syllable frequency to include
    )

    kpms.visualize_transition_bigram(
        project_dir,
        model_name,
        groups,
        trans_mats,
        syll_include,
        normalize=normalize,
        show_syllable_names=True,  # label syllables by index (False) or index and name (True)
    )

    kpms.plot_transition_graph_group(
        project_dir,
        model_name,
        groups,
        trans_mats,
        usages,
        syll_include,
        layout="spring",  # transition graph layout ("circular" or "spring")
        show_syllable_names=False,  # label syllables by index (False) or index and name (True)
    )

    kpms.plot_transition_graph_difference(
        project_dir, model_name, groups, trans_mats, usages, syll_include, layout="spring"
    )


def get_session_data(project_dir, model_name):
    index_filepath = os.path.join(project_dir, "index.csv")
    if os.path.exists(index_filepath):
        index_data = pd.read_csv(index_filepath, index_col=False)
    else:
        print("index.csv não encontrado")
        exit()

    results = kpms.load_results(project_dir, model_name)

    sessions = []

    for k, v in results.items():
        n_frame = v["centroid"].shape[0]
        sessions.append([index_data[index_data["name"] == k]["session"].values[0]] * n_frame)

    return sessions
