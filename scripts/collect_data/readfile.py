import h5py

# Open HDF5 file
# file_path = "./data/groundtruth/episode_20260309_165648.hdf5"

# file_path = "./data/groundtruth/episode_20260309_165648.hdf5"

# with h5py.File(file_path, "r") as f:
#     # List top-level groups/datasets
#     print("Keys:", list(f.keys()))

#     # Read a specific dataset
#     dataset = f["action"][100]   # read entire dataset
#     print(dataset)

import h5py

# file_path = "./data/groundtruth/episode_20260309_165648.hdf5"

file_path = "./data/test/episode_20260413_175507.hdf5"

with h5py.File(file_path, "r") as f:
    action = f["action"][:50]   # read all data
    print(action)



# import h5py

# file_path = "./data/groundtruth/episode_20260309_165648.hdf5"

# # file_path = "./data/test/episode_20260323_185317.hdf5"


# def print_hdf5_structure(name, obj):
#     indent = "  " * name.count("/")
#     if isinstance(obj, h5py.Group):
#         print(f"{indent}[Group] {name}")
#     elif isinstance(obj, h5py.Dataset):
#         print(f"{indent}[Dataset] {name} shape={obj.shape} dtype={obj.dtype}")
    
#     # print attributes
#     for key, value in obj.attrs.items():
#         print(f"{indent}  - attr[{key}] = {value}")

# with h5py.File(file_path, "r") as f:
#     print("=== HDF5 Structure ===")
#     f.visititems(print_hdf5_structure)




# === HDF5 Structure ===
# [Dataset] action shape=(477, 7) dtype=float32
# [Group] observations
#   [Dataset] observations/O_F_ext_hat_K shape=(477, 6) dtype=float32
#   [Group] observations/depth
#   [Dataset] observations/dtau_J shape=(477, 7) dtype=float32
#   [Dataset] observations/ee_pos_q shape=(477, 4) dtype=float32
#   [Dataset] observations/ee_pos_rpy shape=(477, 3) dtype=float32
#   [Dataset] observations/ee_pos_t shape=(477, 3) dtype=float32
#   [Dataset] observations/ee_twist_ang shape=(477, 3) dtype=float32
#   [Dataset] observations/ee_twist_lin shape=(477, 3) dtype=float32
#   [Dataset] observations/elbow_jnt3_pos shape=(477, 1) dtype=float32
#   [Dataset] observations/elbow_jnt4_flip shape=(477, 1) dtype=float32
#   [Dataset] observations/gpos shape=(477, 1) dtype=float32
#   [Group] observations/images
#     [Dataset] observations/images/ext1 shape=(477, 480, 640, 3) dtype=uint8
#     [Dataset] observations/images/wrist shape=(477, 480, 640, 3) dtype=uint8
#   [Dataset] observations/qpos shape=(477, 7) dtype=float32
#   [Dataset] observations/qvel shape=(477, 7) dtype=float32
#   [Dataset] observations/tau_J shape=(477, 7) dtype=float32
#   [Dataset] observations/tau_ext_hat_filtered shape=(477, 7) dtype=float32
# [Dataset] tm shape=(477, 1) dtype=float32



# === HDF5 Structure ===
# [Dataset] action shape=(111, 7) dtype=float32
# [Group] observations
#   [Dataset] observations/O_F_ext_hat_K shape=(111, 6) dtype=float32
#   [Group] observations/depth
#   [Dataset] observations/dtau_J shape=(111, 7) dtype=float32
#   [Dataset] observations/ee_pos_q shape=(111, 4) dtype=float32
#   [Dataset] observations/ee_pos_rpy shape=(111, 3) dtype=float32
#   [Dataset] observations/ee_pos_t shape=(111, 3) dtype=float32
#   [Dataset] observations/ee_twist_ang shape=(111, 3) dtype=float32
#   [Dataset] observations/ee_twist_lin shape=(111, 3) dtype=float32
#   [Dataset] observations/elbow_jnt3_pos shape=(111, 1) dtype=float32
#   [Dataset] observations/elbow_jnt4_flip shape=(111, 1) dtype=float32
#   [Dataset] observations/gpos shape=(111, 1) dtype=float32
#   [Group] observations/images
#     [Dataset] observations/images/ext1 shape=(111, 480, 640, 3) dtype=uint8
#     [Dataset] observations/images/wrist shape=(111, 480, 640, 3) dtype=uint8
#   [Dataset] observations/qpos shape=(111, 7) dtype=float32
#   [Dataset] observations/qvel shape=(111, 7) dtype=float32
#   [Dataset] observations/tau_J shape=(111, 7) dtype=float32
#   [Dataset] observations/tau_ext_hat_filtered shape=(111, 7) dtype=float32
# [Dataset] tm shape=(111, 1) dtype=float32