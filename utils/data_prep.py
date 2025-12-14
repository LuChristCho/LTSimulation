import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import subprocess

# Configuration
DATASET_NAME = "WaterDrop"
OUTPUT_DIR = os.path.join("temp", "datasets", DATASET_NAME)
BASE_URL = f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}"

def download_data():
    """Downloads the dataset files if they don't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = ["metadata.json", "test.tfrecord", "train.tfrecord", "valid.tfrecord"]
    for file in files:
        file_path = os.path.join(OUTPUT_DIR, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            subprocess.run(["wget", "-O", file_path, f"{BASE_URL}/{file}"], check=True)

def parse_proto(example_proto):
    """Parses a single TFRecord example."""
    context_features = {
        'particle_type': tf.io.FixedLenFeature([], tf.string),
        'key': tf.io.FixedLenFeature([], tf.int64),
    }
    sequence_features = {
        'position': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True)
    }

    context, sequence = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    particle_type = tf.io.decode_raw(context['particle_type'], tf.int64)
    position_flat = tf.map_fn(
        lambda x: tf.io.decode_raw(x, tf.float32),
        sequence['position'],
        dtype=tf.float32
    )

    num_particles = tf.shape(particle_type)[0]
    position = tf.reshape(position_flat, [-1, num_particles, 2])

    return position, particle_type

def update_stats(prev_mean, prev_m2, prev_count, new_data):
    """Updates running statistics (mean and variance)."""
    new_data = new_data.reshape(-1, new_data.shape[-1])
    n = new_data.shape[0]

    if n == 0:
        return prev_mean, prev_m2, prev_count

    new_count = prev_count + n
    delta = new_data - prev_mean
    new_mean = prev_mean + np.sum(delta, axis=0) / new_count
    delta2 = new_data - new_mean
    new_m2 = prev_m2 + np.sum(delta * delta2, axis=0)

    return new_mean, new_m2, new_count

def estimate_total_records(filepath):
    file_size = os.path.getsize(filepath)
    for record in tf.data.TFRecordDataset(filepath).take(1):
        record_size = len(record.numpy()) + 16
        if record_size == 0: return None
        return int(file_size / record_size)
    return None

def convert_split(split_name, input_path, output_dir):
    print(f"Converting split: {split_name}")
    pos_file = os.path.join(output_dir, f"{split_name}_position.dat")
    type_file = os.path.join(output_dir, f"{split_name}_particle_type.dat")
    offset_file = os.path.join(output_dir, f"{split_name}_offset.json")

    offsets = {}
    global_pos_index = 0
    global_type_index = 0
    traj_idx = 0

    vel_mean, vel_m2, vel_count = np.zeros(2), np.zeros(2), 0
    acc_mean, acc_m2, acc_count = np.zeros(2), np.zeros(2), 0

    total_est = estimate_total_records(input_path)

    with open(pos_file, "wb") as f_pos, open(type_file, "wb") as f_type:
        dataset = tf.data.TFRecordDataset(input_path)
        for raw_record in tqdm(dataset, unit="traj", total=total_est, desc=f"{split_name}"):
            positions, particle_types = parse_proto(raw_record)
            
            positions = positions.numpy().astype(np.float32)
            particle_types = particle_types.numpy().astype(np.int64)

            f_pos.write(positions.tobytes())
            f_type.write(particle_types.tobytes())

            offsets[traj_idx] = {
                "particle_type": {"offset": global_type_index, "length": len(particle_types)},
                "position": {"offset": global_pos_index, "shape": positions.shape}
            }

            global_pos_index += positions.size
            global_type_index += particle_types.size
            traj_idx += 1

            vel = positions[1:] - positions[:-1]
            vel_mean, vel_m2, vel_count = update_stats(vel_mean, vel_m2, vel_count, vel)

            acc = vel[1:] - vel[:-1]
            acc_mean, acc_m2, acc_count = update_stats(acc_mean, acc_m2, acc_count, acc)

    with open(offset_file, "w") as f:
        json.dump(offsets, f)

    vel_std = np.sqrt(vel_m2 / vel_count)
    acc_std = np.sqrt(acc_m2 / acc_count)
    print(f"Finished split {split_name}")
    return vel_mean, vel_std, acc_mean, acc_std

if __name__ == "__main__":
    download_data()
    
    METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
    splits = {
        "train": os.path.join(OUTPUT_DIR, "train.tfrecord"),
        "test": os.path.join(OUTPUT_DIR, "test.tfrecord"),
        "valid": os.path.join(OUTPUT_DIR, "valid.tfrecord"),
    }

    final_stats = {}
    for split, path in splits.items():
        stats = convert_split(split, path, OUTPUT_DIR)
        final_stats[split] = {
            "vel_mean": stats[0].tolist(), "vel_std": stats[1].tolist(),
            "acc_mean": stats[2].tolist(), "acc_std": stats[3].tolist()
        }

    # Update metadata
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    
    meta["vel_mean"] = final_stats["train"]["vel_mean"]
    meta["vel_std"] = final_stats["train"]["vel_std"]
    meta["acc_mean"] = final_stats["train"]["acc_mean"]
    meta["acc_std"] = final_stats["train"]["acc_std"]
    if "default_connectivity_radius" not in meta:
        meta["default_connectivity_radius"] = 0.015

    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=4)
    print("Data preparation complete.")