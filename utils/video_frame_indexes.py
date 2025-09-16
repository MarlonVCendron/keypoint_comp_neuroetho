import numpy as np

start_frames_to_skip = 120
end_frames_to_skip = 0

def get_video_frame_indexes(coordinates, confidences):
  bounds = {k: (start_frames_to_skip, len(coords) - end_frames_to_skip) for k,coords in coordinates.items()}

  coordinates = {k: coords[bounds[k][0]:bounds[k][1]] for k,coords in coordinates.items()}
  confidences = {k: confs[bounds[k][0]:bounds[k][1]] for k,confs in confidences.items()}
  video_frame_indexes = {k : np.arange(bounds[k][0], bounds[k][1]) for k in bounds}
  return coordinates, confidences, video_frame_indexes
