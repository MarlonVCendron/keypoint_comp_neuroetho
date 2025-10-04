# From https://keypoint-moseq.readthedocs.io/en/latest/_modules/keypoint_moseq/util.html#find_medoid_distance_outliers

import keypoint_moseq as kpms  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os

def get_distance_to_medoid(coordinates: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance from each keypoint to the medoid (median position)
    of all keypoints at each frame.

    Parameters
    -------
    coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Keypoint coordinates where keypoint_dim is 2 or 3.

    Returns
    -------
    distances: ndarray of shape (n_frames, n_keypoints)
        Euclidean distances from each keypoint to the medoid position at each frame.
    """
    medoids = np.median(coordinates, axis=1)  # (n_frames, keypoint_dim)
    return np.linalg.norm(coordinates - medoids[:, None, :], axis=-1)  # (n_frames, n_keypoints)


def find_medoid_distance_outliers(
    coordinates: np.ndarray, outlier_scale_factor: float, **kwargs
) -> dict[str, np.ndarray]:
    """Identify keypoint distance outliers using Median Absolute Deviation (MAD).

    Keypoints are considered outliers when their distance to the medoid at a given timepoint differs
    from its median value by a multiple of the median absolute deviation (MAD) for that keypoint.

    Parameters
    -------
    coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Keypoint coordinates where keypoint_dim is 2 or 3. Only the first two dimensions (x, y) are
        used for distance calculations.

    outlier_scale_factor: float, default=6.0
        Multiplier used to set the outlier threshold. Higher values result in fewer outliers.

    **kwargs
        Additional keyword arguments (ignored), usually overflow from **config().

    Returns
    -------
    result: dict with the following items

        mask: ndarray of shape (n_frames, n_keypoints)
            Boolean array where True indicates outlier keypoints.

        thresholds: ndarray of shape (n_keypoints,)
            Distance thresholds used to classify outlier timepoints for each keypoint.
    """
    distances = get_distance_to_medoid(coordinates)  # (n_frames, n_keypoints)
    medians = np.median(distances, axis=0)  # (n_keypoints,)
    MADs = np.median(np.abs(distances - medians[None, :]), axis=0)  # (n_keypoints,)
    outlier_thresholds = MADs * outlier_scale_factor + medians  # (n_keypoints,)
    outlier_mask = distances > outlier_thresholds[None, :]  # (n_frames, n_keypoints)
    return {"mask": outlier_mask, "thresholds": outlier_thresholds}

def plot_keypoint_traces(
    traces: list[np.ndarray],
    plot_title: Optional[str] = None,
    bodyparts: Optional[list[str]] = None,
    line_labels: Optional[list[str]] = None,
    thresholds: Optional[np.ndarray] = None,
    shading_mask: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Create a multi-panel plot showing keypoint traces over time (used to visualize outliers).

    Creates a figure with one subplot per keypoint, where each subplot shows multiple
    trace lines over time. Optional features include threshold lines, shaded regions,
    and custom labels.

    Parameters
    -------
    traces: list of ndarrays
        List of arrays, each with shape (n_frames, n_keypoints). Each array
        represents a different trace to plot.

    plot_title: str, optional
        Title to display at the top of the figure.

    bodyparts: list of str, optional
        Names of bodyparts corresponding to each keypoint. Used for subplot titles.
        Must have length equal to n_keypoints.

    line_labels: list of str, optional
        Labels for each trace line in the legend. Must have length equal to
        the number of traces.

    thresholds: ndarray of shape (n_keypoints,), optional
        Threshold values to plot as horizontal dashed lines for each keypoint.

    shading_mask: ndarray of shape (n_frames, n_keypoints), optional
        Boolean mask indicating frames to shade (e.g., outlier frames).
        True values will be shaded in grey.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The created figure with subplots for each keypoint.
    """

    if not traces:
        raise ValueError("traces cannot be empty")

    n_keypoints = traces[0].shape[1]

    for i, trace_array in enumerate(traces):
        if trace_array.shape[1] != n_keypoints:
            raise ValueError(
                f"All trace arrays must have same number of keypoints. "
                f"Array {i} has {trace_array.shape[1]} keypoints, expected {n_keypoints}"
            )

    if bodyparts is not None and len(bodyparts) != n_keypoints:
        raise ValueError(
            f"Length of bodyparts list ({len(bodyparts)}) does not match "
            f"number of keypoints in traces ({n_keypoints})"
        )

    if shading_mask is not None:
        if shading_mask.shape != traces[0].shape:
            raise ValueError(
                f"Shading mask shape {shading_mask.shape} must match traces shape {traces[0].shape}"
            )

    fig, axes = plt.subplots(n_keypoints, 1, figsize=(16, 3 * n_keypoints), constrained_layout=True)
    if n_keypoints == 1:
        axes = [axes]  # Ensure axes is always a list

    for keypoint_idx in range(n_keypoints):
        ax = axes[keypoint_idx]

        if shading_mask is not None:
            shaded_frames = np.where(shading_mask[:, keypoint_idx])[0]
            if len(shaded_frames) > 0:
                for frame in shaded_frames:
                    ax.axvspan(frame - 0.5, frame + 0.5, alpha=0.1, color="grey")

        for line_idx, trace_array in enumerate(traces):
            label = line_labels[line_idx] if line_labels else f"Line {line_idx}"
            ax.plot(trace_array[:, keypoint_idx], label=label)

        if thresholds is not None:
            threshold_value = thresholds[keypoint_idx]
            ax.axhline(
                y=threshold_value,
                color="black",
                linestyle="--",
                alpha=0.7,
                label=f"Threshold ({threshold_value:.2f})",
            )

        ax.set_xlabel("Frame")
        ax.set_ylabel("Trace Value")

        if bodyparts is not None:
            ax.set_title(f"{bodyparts[keypoint_idx]}")
        else:
            ax.set_title(f"Keypoint {keypoint_idx}")

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)
    return fig


def plot_medoid_distance_outliers(
    project_dir: str,
    recording_name: str,
    original_coordinates: np.ndarray,
    interpolated_coordinates: np.ndarray,
    outlier_mask,
    outlier_thresholds,
    bodyparts: list[str],
    **kwargs,
):
    """Create and save a plot comparing distance-to-medoid for original vs. interpolated keypoints.

    Generates a multi-panel plot showing the distance from each keypoint to the medoid
    position for both original and interpolated coordinates. The plot includes threshold
    lines and shaded regions for outlier frames. Saves the figure to the QA plots
    directory.

    Parameters
    -------
    project_dir: str
        Path to the project directory where the plot will be saved.

    recording_name: str
        Name of the recording, used for the plot title and filename.

    original_coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Original keypoint coordinates before interpolation.

    interpolated_coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Keypoint coordinates after interpolation.

    outlier_mask: ndarray of shape (n_frames, n_keypoints)
        Boolean mask indicating outlier keypoints (True = outlier).

    outlier_thresholds: ndarray of shape (n_keypoints,)
        Distance thresholds for each keypoint above which points are considered outliers.

    bodyparts: list of str
        Names of bodyparts corresponding to each keypoint. Must have length equal to
        n_keypoints.

    **kwargs
        Additional keyword arguments (ignored), usually overflow from **config().

    Returns
    -------
    None
        The plot is saved to 'QA/plots/keypoint_distance_outliers/{recording_name}.png'.
    """

    plot_path = os.path.join(
        project_dir,
        "quality_assurance",
        "plots",
        "keypoint_distance_outliers",
        f"{recording_name}.png",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    original_distances = get_distance_to_medoid(original_coordinates)  # (n_frames, n_keypoints)
    interpolated_distances = get_distance_to_medoid(
        interpolated_coordinates
    )  # (n_frames, n_keypoints)

    fig = plot_keypoint_traces(
        traces=[original_distances, interpolated_distances],
        plot_title=recording_name,
        bodyparts=bodyparts,
        line_labels=["Original", "Interpolated"],
        thresholds=outlier_thresholds,
        shading_mask=outlier_mask,
    )

    fig.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved keypoint distance outlier plot for {recording_name} to {plot_path}.")

def filter_outliers(coordinates: np.ndarray, confidences: np.ndarray, config: dict, cb: callable = None) -> tuple[np.ndarray, np.ndarray]:
    for recording_name in coordinates:
        raw_coords = coordinates[recording_name].copy()
        outliers = find_medoid_distance_outliers(raw_coords, outlier_scale_factor=6.0, **config)
        coordinates[recording_name] = kpms.interpolate_keypoints(raw_coords, outliers["mask"])
        confidences[recording_name] = np.where(outliers["mask"], 0, confidences[recording_name])
        if cb is not None:
            cb(coordinates, confidences, outliers, recording_name, raw_coords)
    return coordinates, confidences, outliers