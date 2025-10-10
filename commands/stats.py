import keypoint_moseq as kpms  # type: ignore
import os
import pandas as pd

from utils.load_data_and_config import load_config

min_frequency = 0.005


def stats(project_dir, model_name):
    config = load_config(project_dir)
    save_dir = os.path.join(project_dir, model_name)

    # index.csv para rotular os grupos
    kpms.interactive_group_setting(project_dir, model_name)

    if os.path.exists(os.path.join(save_dir, "moseq_df.csv")):
        moseq_df = pd.read_csv(os.path.join(save_dir, "moseq_df.csv"))
    else:
        moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
        moseq_df.to_csv(os.path.join(save_dir, "moseq_df.csv"), index=False)
        print("Saved `moseq_df` dataframe to", save_dir)

    if os.path.exists(os.path.join(save_dir, "stats_df.csv")):
        stats_df = pd.read_csv(os.path.join(save_dir, "stats_df.csv"))
    else:
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
    kpms.label_syllables(project_dir, model_name, moseq_df)

    kpms.plot_syll_stats_with_sem(
        stats_df,
        project_dir,
        model_name,
        plot_sig=True,  # whether to mark statistical significance with a star
        thresh=0.05,  # significance threshold
        stat="frequency",  # statistic to be plotted (e.g. 'duration' or 'velocity_px_s_mean')
        order="stat",  # order syllables by overall frequency ("stat") or degree of difference ("diff")
        ctrl_group="a",  # name of the control group for statistical testing
        exp_group="b",  # name of the experimental group for statistical testing
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
        layout="circular",  # transition graph layout ("circular" or "spring")
        show_syllable_names=False,  # label syllables by index (False) or index and name (True)
    )

    kpms.plot_transition_graph_difference(
        project_dir, model_name, groups, trans_mats, usages, syll_include, layout="circular"
    )
