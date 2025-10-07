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


def get_keypoint_velocities(coordinates: np.ndarray, fps: float = 30.0) -> np.ndarray:
    displacement = np.diff(coordinates, axis=0)  # (n_frames-1, n_keypoints, keypoint_dim)

    velocities = np.linalg.norm(displacement, axis=-1) * fps

    return velocities


def get_keypoint_to_keypoint_distances(coordinates: np.ndarray) -> np.ndarray:
    n_frames, n_keypoints, keypoint_dim = coordinates.shape
    
    # Broadcasting
    diff = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    
    distances = np.linalg.norm(diff, axis=-1)
    
    return distances


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


def find_velocity_outliers(
    coordinates: np.ndarray, outlier_scale_factor: float = 4.0, fps: float = 30.0, **kwargs
) -> dict[str, np.ndarray]:
    velocities = get_keypoint_velocities(coordinates, fps)

    medians = np.median(velocities, axis=0)
    MADs = np.median(np.abs(velocities - medians[None, :]), axis=0)

    outlier_thresholds = MADs * outlier_scale_factor + medians

    outlier_mask = velocities > outlier_thresholds[None, :]

    return {"mask": outlier_mask, "thresholds": outlier_thresholds}


# Se o keypoint for anormalmente distante de X% dos outros keypoints, ele é considerado um outlier.
def find_keypoint_distance_outliers(
    coordinates: np.ndarray,
    outlier_scale_factor: float = 6.0,
    outlier_threshold_percentage: float = 0.5, #X%
    **kwargs,
) -> dict[str, np.ndarray]:
    distances = get_keypoint_to_keypoint_distances(
        coordinates
    )  # (n_frames, n_keypoints, n_keypoints)
    n_frames, n_keypoints, _ = distances.shape

    medians = np.median(distances, axis=0)  # (n_keypoints, n_keypoints)
    MADs = np.median(np.abs(distances - medians[None, :, :]), axis=0)  # (n_keypoints, n_keypoints)

    outlier_thresholds = MADs * outlier_scale_factor + medians  # (n_keypoints, n_keypoints)

    distance_outlier_mask = (
        distances > outlier_thresholds[None, :, :]
    )  # (n_frames, n_keypoints, n_keypoints)

    keypoint_outlier_mask = np.zeros((n_frames, n_keypoints), dtype=bool)

    for f in range(n_frames):
        for i in range(n_keypoints):
            distances_from_i = distance_outlier_mask[f, i, :]  # (n_keypoints,)

            outlier_count = np.sum(
                distances_from_i
            )
            total_other_keypoints = n_keypoints - 1

            if outlier_count >= outlier_threshold_percentage * total_other_keypoints:
                keypoint_outlier_mask[f, i] = True

    return {
        "mask": keypoint_outlier_mask,
        "thresholds": outlier_thresholds,
        "distance_outlier_mask": distance_outlier_mask,
    }


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


def plot_velocity_outliers(
    project_dir: str,
    recording_name: str,
    original_coordinates: np.ndarray,
    interpolated_coordinates: np.ndarray,
    outlier_mask: np.ndarray,
    outlier_thresholds: np.ndarray,
    bodyparts: list[str],
    fps: float = 30.0,
    **kwargs,
):
    plot_path = os.path.join(
        project_dir,
        "quality_assurance",
        "plots",
        "keypoint_velocity_outliers",
        f"{recording_name}.png",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    original_velocities = get_keypoint_velocities(
        original_coordinates, fps
    )  # (n_frames-1, n_keypoints)
    interpolated_velocities = get_keypoint_velocities(
        interpolated_coordinates, fps
    )  # (n_frames-1, n_keypoints)

    fig = plot_keypoint_traces(
        traces=[original_velocities, interpolated_velocities],
        plot_title=f"{recording_name} - Velocity Outliers",
        bodyparts=bodyparts,
        line_labels=["Original", "Interpolated"],
        thresholds=outlier_thresholds,
        shading_mask=outlier_mask,
    )

    fig.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved keypoint velocity outlier plot for {recording_name} to {plot_path}.")


def plot_keypoint_distance_outliers(
    project_dir: str,
    recording_name: str,
    original_coordinates: np.ndarray,
    interpolated_coordinates: np.ndarray,
    outlier_mask: np.ndarray,
    outlier_thresholds: np.ndarray,
    bodyparts: list[str],
    **kwargs,
):
    plot_path = os.path.join(
        project_dir,
        "quality_assurance",
        "plots",
        "keypoint_distance_outliers",
        f"{recording_name}.png",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Calculate keypoint-to-keypoint distances
    original_distances = get_keypoint_to_keypoint_distances(original_coordinates)
    interpolated_distances = get_keypoint_to_keypoint_distances(interpolated_coordinates)

    n_frames, n_keypoints, _ = original_distances.shape

    # Create a figure with subplots for each keypoint
    fig, axes = plt.subplots(n_keypoints, 1, figsize=(16, 3 * n_keypoints), constrained_layout=True)
    if n_keypoints == 1:
        axes = [axes]

    for keypoint_idx in range(n_keypoints):
        ax = axes[keypoint_idx]

        # Shade outlier frames
        shaded_frames = np.where(outlier_mask[:, keypoint_idx])[0]
        if len(shaded_frames) > 0:
            for frame in shaded_frames:
                ax.axvspan(frame - 0.5, frame + 0.5, alpha=0.1, color="grey")

        # Plot distances from this keypoint to all other keypoints
        for other_keypoint_idx in range(n_keypoints):
            if other_keypoint_idx != keypoint_idx:
                # Original distances
                ax.plot(
                    original_distances[:, keypoint_idx, other_keypoint_idx],
                    alpha=0.7,
                    label=f"Original to {bodyparts[other_keypoint_idx]}",
                )

                # Interpolated distances
                ax.plot(
                    interpolated_distances[:, keypoint_idx, other_keypoint_idx],
                    alpha=0.7,
                    linestyle="--",
                    label=f"Interpolated to {bodyparts[other_keypoint_idx]}",
                )

                # Threshold line
                threshold_value = outlier_thresholds[keypoint_idx, other_keypoint_idx]
                ax.axhline(y=threshold_value, color="black", linestyle=":", alpha=0.5)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance (pixels)")
        ax.set_title(f"Distances from {bodyparts[keypoint_idx]} to other keypoints")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{recording_name} - Keypoint Distance Outliers", fontsize=16)
    fig.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved keypoint distance outlier plot for {recording_name} to {plot_path}.")


def filter_outliers(
    coordinates: np.ndarray,
    confidences: np.ndarray,
    config: dict,
    cb: callable = None,
    use_keypoint_distance_outliers: bool = True,
    keypoint_distance_outlier_scale_factor: float = 4.0,
    keypoint_distance_outlier_threshold_percentage: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    for i, recording_name in enumerate(coordinates):
        print(f"{i+1}/{len(coordinates)}: {recording_name}")
        raw_coords = coordinates[recording_name].copy()

        medoid_outliers = find_medoid_distance_outliers(
            raw_coords, outlier_scale_factor=4.0, **config
        )

        if use_keypoint_distance_outliers:
            keypoint_distance_outliers = find_keypoint_distance_outliers(
                raw_coords,
                outlier_scale_factor=keypoint_distance_outlier_scale_factor,
                outlier_threshold_percentage=keypoint_distance_outlier_threshold_percentage,
                **config,
            )

            # Combine outlier masks (OR operation - if either method detects outlier, mark as outlier)
            combined_mask = medoid_outliers["mask"] | keypoint_distance_outliers["mask"]

            # Create combined outliers dict
            combined_outliers = {
                "mask": combined_mask,
                "medoid_thresholds": medoid_outliers["thresholds"],
                "keypoint_distance_thresholds": keypoint_distance_outliers["thresholds"],
                "medoid_outliers": medoid_outliers,
                "keypoint_distance_outliers": keypoint_distance_outliers,
            }
        else:
            combined_outliers = medoid_outliers

        coordinates[recording_name] = kpms.interpolate_keypoints(
            raw_coords, combined_outliers["mask"]
        )
        confidences[recording_name] = np.where(
            combined_outliers["mask"], 0, confidences[recording_name]
        )

        if cb is not None:
            cb(coordinates, confidences, combined_outliers, recording_name, raw_coords)

    return coordinates, confidences, combined_outliers
