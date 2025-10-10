import h5py

results_old = '/home/marlon/edu/mestrado/comp_neuroetho/keypoint_comp_neuroetho/projects/elm_ms/9k/results.h5.new'
results_new = '/home/marlon/edu/mestrado/comp_neuroetho/keypoint_comp_neuroetho/projects/elm_ms/9k/results.h5.old'
results = '/home/marlon/edu/mestrado/comp_neuroetho/keypoint_comp_neuroetho/projects/elm_ms/9k/results.h5'

def copy_group(source_group, target_group):
    """Recursively copy HDF5 groups and datasets with their attributes"""
    for key in source_group.keys():
        if isinstance(source_group[key], h5py.Group):
            new_group = target_group.create_group(key)
            # Copy group attributes
            for attr_name, attr_value in source_group[key].attrs.items():
                new_group.attrs[attr_name] = attr_value
            copy_group(source_group[key], new_group)
        else:
            # Copy dataset with attributes
            source_group.copy(key, target_group)
            # Copy dataset attributes
            for attr_name, attr_value in source_group[key].attrs.items():
                target_group[key].attrs[attr_name] = attr_value

with h5py.File(results, 'w') as f_result:
    with h5py.File(results_old, 'r') as f_old:
        copy_group(f_old, f_result)
    
    with h5py.File(results_new, 'r') as f_new:
        copy_group(f_new, f_result)