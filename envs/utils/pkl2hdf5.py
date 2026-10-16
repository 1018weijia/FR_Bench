import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # If this is a leaf, append the data directly
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # If this is a nested dict, process it recursively
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            # Try to convert to a numpy array
            try:
                value_array = np.array(value)
                
                # Check whether conversion succeeded (no object dtype or irregular shape)
                if value_array.dtype == object:
                    # Irregular array; convert to a list of strings
                    raise ValueError("Irregular array")
                
                if "rgb" in key:
                    encode_data, max_len = images_encoding(value_array)
                    hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
                else:
                    hdf5_group.create_dataset(key, data=value_array)
                    
            except (ValueError, Exception) as e:
                # Conversion failed or the array is irregular; try another approach
                try:
                    # Try saving as a list of strings
                    value_str = [str(v) for v in value]
                    hdf5_group.create_dataset(key, data=value_str, dtype=h5py.string_dtype())
                except Exception as e2:
                    # Give up and skip
                    print(f"Warning: skipping key '{key}' that cannot be saved: {e2}")
        else:
            # Handle a single value
            try:
                # Try saving directly
                hdf5_group.create_dataset(key, data=value)
            except Exception:
                # If that fails, convert to a string
                try:
                    if isinstance(value, str):
                        hdf5_group.create_dataset(key, data=value, dtype=h5py.string_dtype())
                    else:
                        hdf5_group.create_dataset(key, data=str(value), dtype=h5py.string_dtype())
                except Exception as e:
                    print(f"Warning: skipping key '{key}' that cannot be saved: {e}")


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    # Try to generate a video; skip if there is no rgb data
    try:
        if "observation" in data_list and "head_camera" in data_list["observation"] and "rgb" in data_list["observation"]["head_camera"]:
            images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path)
        else:
            print("Warning: rgb data not found, skipping video generation")
    except Exception as e:
        print(f"Error while generating video: {e}")

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)
