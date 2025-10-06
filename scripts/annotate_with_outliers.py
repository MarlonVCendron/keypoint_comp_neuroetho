# Gerado por IA. Gera dois videos com os keypoints original e com outliers filtrados.

import argparse
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import keypoint_moseq as kpms
from typing import Optional, Tuple, Dict, Any

# Add the parent directory to the path to import our utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.find_medoid_distance_outliers import (
    find_medoid_distance_outliers, 
    find_keypoint_distance_outliers,
    get_distance_to_medoid
)


def load_dlc_csv(csv_path):
    """
    Returns:
      coords: numpy array shape (n_frames, n_keypoints, 3) -> (x, y, likelihood)
      names: list of keypoint names length n_keypoints
    This tries two common DLC CSV formats:
      1) MultiIndex header like (scorer, bodypart, coords) where coords are x,y,likelihood
      2) Flat header with columns like "nose_x", "nose_y", "nose_likelihood" or "nose x","nose y"
    """
    # try MultiIndex header first
    try:
        df = pd.read_csv(csv_path, header=[0,1,2], index_col=0)
        # columns levels: level0=scorer, level1=bodypart, level2=coord (x,y,likelihood)
        # build ordered list of bodyparts from level 1 unique preserving order
        bodyparts = []
        for _, bp, _ in df.columns:
            if bp not in bodyparts:
                bodyparts.append(bp)
        n_frames = df.shape[0]
        coords = []
        for bp in bodyparts:
            # access columns for this bodypart
            x = df.xs((slice(None), bp, 'x'), axis=1, drop_level=False)
            # simpler access:
            try:
                xs = df.xs((slice(None), bp, 'x'), axis=1, level=(0,1,2)).squeeze()
                ys = df.xs((slice(None), bp, 'y'), axis=1, level=(0,1,2)).squeeze()
                ls = df.xs((slice(None), bp, 'likelihood'), axis=1, level=(0,1,2)).squeeze()
            except Exception:
                # fallback: try 'score' or 'likelihood' synonyms
                ls = df.xs((slice(None), bp, 'likelihood'), axis=1, level=(0,1,2)).squeeze()
                xs = df.xs((slice(None), bp, 'x'), axis=1, level=(0,1,2)).squeeze()
                ys = df.xs((slice(None), bp, 'y'), axis=1, level=(0,1,2)).squeeze()
            coords.append(np.vstack((xs.values, ys.values, ls.values)).T)  # (n_frames, 3)
        coords = np.stack(coords, axis=1)  # (n_frames, n_kp, 3)
        return coords, bodyparts
    except Exception:
        # fallback: flat csv
        df = pd.read_csv(csv_path, header=0)
        cols = list(df.columns)
        # try patterns: name_x, name_y, name_likelihood or "name x", "name y"
        names = []
        for c in cols:
            if c.endswith('_x'):
                names.append(c[:-2])
            elif c.endswith(' x'):
                names.append(c[:-2])
        names = list(dict.fromkeys(names))  # preserve order, unique
        if not names:
            # try to detect columns grouped as [name, name.1, name.2,...]
            # or columns like "scorer ; bodypart ; x" uncommon for flat CSVs
            # As a last resort, try columns in groups of 3: (x,y,likelihood)
            ncols = len(cols)
            if ncols % 3 == 0:
                n_kp = ncols // 3
                # assume names = col0_base, col3_base etc
                names = [f'kp{i}' for i in range(n_kp)]
                arr = df.values
                # reshape to (n_frames, n_kp, 3)
                coords = arr.reshape(len(df), n_kp, 3)
                return coords, names
            else:
                raise RuntimeError("Could not parse CSV header. Please send the first few lines of the CSV if you want help.")
        # build coords array
        frames = len(df)
        all_kp = []
        for n in names:
            xcol = None
            ycol = None
            lcol = None
            if n + '_x' in df.columns:
                xcol = n + '_x'; ycol = n + '_y'; lcol = n + '_likelihood' if n + '_likelihood' in df.columns else (n + '_prob' if n + '_prob' in df.columns else None)
            elif n + ' x' in df.columns:
                xcol = n + ' x'; ycol = n + ' y'; lcol = n + ' likelihood' if n + ' likelihood' in df.columns else None
            else:
                # try base columns like 'x' 'y' with repeated blocks - skip
                raise RuntimeError(f"Couldn't find x/y columns for bodypart {n}")
            xs = df[xcol].values
            ys = df[ycol].values
            if lcol is not None:
                ls = df[lcol].values
            else:
                # if no likelihood, set to ones
                ls = np.ones_like(xs, dtype=float)
            all_kp.append(np.vstack((xs, ys, ls)).T)
        coords = np.stack(all_kp, axis=1)  # (n_frames, n_kp, 3)
        return coords, names


def apply_outlier_filtering(coords: np.ndarray, 
                          outlier_scale_factor: float = 6.0,
                          use_keypoint_distance_outliers: bool = True,
                          keypoint_distance_outlier_scale_factor: float = 6.0,
                          keypoint_distance_outlier_threshold_percentage: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply outlier filtering to keypoint coordinates.
    
    Returns:
        filtered_coords: numpy array with outliers interpolated
        outlier_info: dictionary containing outlier masks and statistics
    """
    # Extract x,y coordinates (first 2 dimensions)
    coordinates_xy = coords[:, :, :2]  # (n_frames, n_keypoints, 2)
    
    # Find medoid distance outliers
    medoid_outliers = find_medoid_distance_outliers(
        coordinates_xy, outlier_scale_factor=outlier_scale_factor
    )
    
    if use_keypoint_distance_outliers:
        # Find keypoint-to-keypoint distance outliers
        keypoint_distance_outliers = find_keypoint_distance_outliers(
            coordinates_xy,
            outlier_scale_factor=keypoint_distance_outlier_scale_factor,
            outlier_threshold_percentage=keypoint_distance_outlier_threshold_percentage,
        )
        
        # Combine outlier masks (OR operation)
        combined_mask = medoid_outliers["mask"] | keypoint_distance_outliers["mask"]
        
        outlier_info = {
            "mask": combined_mask,
            "medoid_thresholds": medoid_outliers["thresholds"],
            "keypoint_distance_thresholds": keypoint_distance_outliers["thresholds"],
            "medoid_outliers": medoid_outliers,
            "keypoint_distance_outliers": keypoint_distance_outliers,
        }
    else:
        combined_mask = medoid_outliers["mask"]
        outlier_info = medoid_outliers
    
    # Interpolate outliers using keypoint-moseq
    filtered_coords = kpms.interpolate_keypoints(coordinates_xy, combined_mask)
    
    # Add back the likelihood dimension
    filtered_coords_with_likelihood = np.zeros_like(coords)
    filtered_coords_with_likelihood[:, :, :2] = filtered_coords
    filtered_coords_with_likelihood[:, :, 2] = coords[:, :, 2]  # Keep original likelihood
    
    # Set likelihood to 0 for outlier points
    filtered_coords_with_likelihood[:, :, 2] = np.where(
        combined_mask, 0, filtered_coords_with_likelihood[:, :, 2]
    )
    
    return filtered_coords_with_likelihood, outlier_info


def create_side_by_side_video(video_path: str, 
                             csv_path: str, 
                             out_path: str,
                             likelihood_thresh: float = 0.6,
                             skeleton_pairs: Optional[list] = None,
                             point_radius: int = 4,
                             thickness: int = 2,
                             fps_out: Optional[float] = None,
                             draw_names: bool = False,
                             outlier_scale_factor: float = 6.0,
                             use_keypoint_distance_outliers: bool = True,
                             keypoint_distance_outlier_scale_factor: float = 6.0,
                             keypoint_distance_outlier_threshold_percentage: float = 0.5):
    """
    Create a side-by-side video showing original vs outlier-filtered keypoints.
    """
    print("Loading DLC CSV data...")
    coords, names = load_dlc_csv(csv_path)
    n_frames_csv = coords.shape[0]
    n_kp = coords.shape[1]
    
    print("Applying outlier filtering...")
    filtered_coords, outlier_info = apply_outlier_filtering(
        coords,
        outlier_scale_factor=outlier_scale_factor,
        use_keypoint_distance_outliers=use_keypoint_distance_outliers,
        keypoint_distance_outlier_scale_factor=keypoint_distance_outlier_scale_factor,
        keypoint_distance_outlier_threshold_percentage=keypoint_distance_outlier_threshold_percentage
    )
    
    # Print outlier statistics
    outlier_mask = outlier_info["mask"]
    total_keypoints = n_frames_csv * n_kp
    outlier_count = np.sum(outlier_mask)
    outlier_percentage = (outlier_count / total_keypoints) * 100
    print(f"Outlier detection summary:")
    print(f"  Total keypoints: {total_keypoints:,}")
    print(f"  Outliers detected: {outlier_count:,} ({outlier_percentage:.2f}%)")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps_out is None:
        fps_out = fps
    
    # Create side-by-side output video (double width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps_out, (w * 2, h))
    
    # Process frames
    n_to_process = min(n_frames_csv, total_frames)
    if n_frames_csv != total_frames:
        print(f"Warning: CSV has {n_frames_csv} frames, video has {total_frames} frames. Processing {n_to_process} frames (min).")
    
    # Default skeleton (empty if not provided)
    if skeleton_pairs is None:
        skeleton_pairs = []
    
    # Color function for keypoints
    def kp_color(i):
        hue = int(i * 179 / max(1, n_kp-1)) if n_kp > 1 else 0
        hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        return tuple(int(x) for x in bgr)
    
    # Outlier color (red)
    outlier_color = (0, 0, 255)  # BGR format
    
    pbar = tqdm(total=n_to_process, desc="Creating side-by-side video")
    frame_idx = 0
    
    while frame_idx < n_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create side-by-side frame
        side_by_side = np.zeros((h, w * 2, 3), dtype=np.uint8)
        side_by_side[:, :w] = frame  # Left side: original
        side_by_side[:, w:] = frame  # Right side: filtered
        
        # Get keypoint data for this frame
        original_kp_frame = coords[frame_idx]  # (n_kp, 3)
        filtered_kp_frame = filtered_coords[frame_idx]  # (n_kp, 3)
        frame_outlier_mask = outlier_info["mask"][frame_idx]  # (n_kp,)
        
        # Draw on both sides
        for side_idx, (kp_frame, x_offset) in enumerate([(original_kp_frame, 0), (filtered_kp_frame, w)]):
            # Draw skeleton lines
            for a, b in skeleton_pairs:
                if a in names and b in names:
                    ia = names.index(a)
                    ib = names.index(b)
                    xa, ya, la = kp_frame[ia]
                    xb, yb, lb = kp_frame[ib]
                    
                    if (np.isfinite(xa) and np.isfinite(xb) and 
                        la >= likelihood_thresh and lb >= likelihood_thresh):
                        
                        pt_a = (int(round(xa + x_offset)), int(round(ya)))
                        pt_b = (int(round(xb + x_offset)), int(round(yb)))
                        
                        # Use outlier color if either keypoint is an outlier
                        if side_idx == 0:  # Original side
                            color = kp_color(ia)
                        else:  # Filtered side
                            if frame_outlier_mask[ia] or frame_outlier_mask[ib]:
                                color = outlier_color
                            else:
                                color = kp_color(ia)
                        
                        cv2.line(side_by_side, pt_a, pt_b, color, thickness)
            
            # Draw keypoints
            for i in range(n_kp):
                x, y, l = kp_frame[i]
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                
                if l >= likelihood_thresh:
                    pt = (int(round(x + x_offset)), int(round(y)))
                    
                    # Choose color based on side and outlier status
                    if side_idx == 0:  # Original side
                        color = kp_color(i)
                    else:  # Filtered side
                        if frame_outlier_mask[i]:
                            color = outlier_color
                        else:
                            color = kp_color(i)
                    
                    cv2.circle(side_by_side, pt, point_radius, color, -1)
                    
                    if draw_names:
                        cv2.putText(side_by_side, names[i], 
                                  (pt[0] + 6, pt[1] - 6), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Add labels
        cv2.putText(side_by_side, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_by_side, "Filtered (Red=Outliers)", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(side_by_side)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    print(f"Saved side-by-side comparison video to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side video showing original vs outlier-filtered keypoints")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--csv", required=True, help="Path to DLC CSV file")
    parser.add_argument("--out", default="annotated_comparison.mp4", help="Output video path")
    parser.add_argument("--thresh", type=float, default=0.6, help="Likelihood threshold for keypoints")
    parser.add_argument("--radius", type=int, default=4, help="Keypoint circle radius")
    parser.add_argument("--thickness", type=int, default=2, help="Skeleton line thickness")
    parser.add_argument("--draw-names", action="store_true", help="Draw keypoint names")
    parser.add_argument("--skeleton", default=None, help="Skeleton pairs as comma-separated pairs (e.g., 'nose-neck,neck-Lshoulder')")
    
    # Outlier detection parameters
    parser.add_argument("--outlier-scale-factor", type=float, default=6.0, 
                       help="Scale factor for medoid distance outlier detection")
    parser.add_argument("--no-keypoint-distance-outliers", action="store_true",
                       help="Disable keypoint-to-keypoint distance outlier detection")
    parser.add_argument("--keypoint-distance-scale-factor", type=float, default=6.0,
                       help="Scale factor for keypoint distance outlier detection")
    parser.add_argument("--keypoint-distance-threshold-percentage", type=float, default=0.5,
                       help="Percentage threshold for keypoint distance outliers")
    
    args = parser.parse_args()
    
    skeleton_pairs = None
    if args.skeleton:
        skeleton_pairs = [tuple(p.split('-')) for p in args.skeleton.split(',')]
    
    create_side_by_side_video(
        video_path=args.video,
        csv_path=args.csv,
        out_path=args.out,
        likelihood_thresh=args.thresh,
        skeleton_pairs=skeleton_pairs,
        point_radius=args.radius,
        thickness=args.thickness,
        draw_names=args.draw_names,
        outlier_scale_factor=args.outlier_scale_factor,
        use_keypoint_distance_outliers=not args.no_keypoint_distance_outliers,
        keypoint_distance_outlier_scale_factor=args.keypoint_distance_scale_factor,
        keypoint_distance_outlier_threshold_percentage=args.keypoint_distance_threshold_percentage
    )


if __name__ == "__main__":
    main()
