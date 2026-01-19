import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
)
from PyriteUtility.planning_control.filtering import LiveLPFilter

import pathlib
import shutil
import numpy as np
import pandas as pd
import zarr
import cv2
import concurrent.futures

from scipy.signal import butter, lfilter
import h5py

# check environment variables
if "PYRITE_RAW_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_RAW_DATASET_FOLDERS")
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")


def image_read(rgb_dir, rgb_file_list, i, output_data_rgb, output_data_rgb_time_stamps):
    img_name = rgb_file_list[i]
    img = cv2.imread(str(rgb_dir.joinpath(img_name)))

    if img is None:
        print(f"Error: Failed to read {img_name}")
        return False
    # convert BGR to RGB for imageio
    output_data_rgb[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # time_img_ms = float(img_name[11:22])
    # output_data_rgb_time_stamps[i] = time_img_ms

    try:
        # time_img_ms = float(img_name.split('.')[0])
        time_img_ms = int(img_name.split('.')[0])
        output_data_rgb_time_stamps[i] = time_img_ms
    except ValueError:
        print(f"Error: Could not parse timestamp from {img_name}")
        return False

    return True

def compute_aligned_filtered_wrench(episode_dir, id_list, target_timestamps):
    """
    从 H5 读取 1000Hz 原始数据进行滤波，并对齐到 target_timestamps。
    """
    h5_path = episode_dir.joinpath("lowdim/lowdim_filled.h5")
    
    # 定义滤波器: fs=1000Hz, cutoff=5Hz, order=5
    # nyq = 0.5 * 1000 = 500
    b, a = butter(5, 5 / 500.0, btype='low')
    
    output_filtered_list = []
    
    with h5py.File(h5_path, 'r') as f:
        # 读取全量 1000Hz 数据
        raw_ts = f['timestamp'][:]

        raw_wrench = f['force_torque_062046'][:] 
        
        # 1. 在全量数据上滤波 (保证 fs=1000 稳定)
        filtered_full = lfilter(b, a, raw_wrench, axis=0)
        
        # 2. 建立哈希表 {timestamp: filtered_wrench}
        data_map = {ts: w for ts, w in zip(raw_ts, filtered_full)}
        
        # 3. 基于图片时间戳提取
        for i, _ in enumerate(id_list):
            current_ts_list = target_timestamps[i]
            extracted_wrench = []
            
            for ts in current_ts_list:
                ts_int = int(ts)
                if ts_int in data_map:
                    extracted_wrench.append(data_map[ts_int])
                else:
                    # 查不到时：打印警告 + 补零
                    print(f"Warning: Timestamp {ts_int} not found in filtered H5 data! Filling with zeros.")
                    extracted_wrench.append(np.zeros(6))
            
            output_filtered_list.append(np.array(extracted_wrench))
            
    return output_filtered_list


# specify the input and output directories
id_list = [0]  # single robot
# id_list = [0, 1] # bimanual

input_dir = pathlib.Path(
    os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/flip_up_new_v5"
)
output_dir = pathlib.Path(os.environ.get("PYRITE_DATASET_FOLDERS") + "/flip_up_new_v5")

robot_timestamp_dir = output_dir.joinpath("robot_timestamp")
wrench_timestamp_dir = output_dir.joinpath("wrench_timestamp")
rgb_timestamp_dir = output_dir.joinpath("rgb_timestamp")

# clean and create output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)


# # check for black images
# def check_black_images(rgb_file_list, rgb_dir, i, prefix):
#     f = rgb_file_list[i]
#     img = cv2.imread(str(rgb_dir.joinpath(f)))
#     # print the mean of the image
#     img_mean = np.mean(img)
#     if img_mean < 50:
#         print(f"{prefix}, {f} has mean value of {img_mean}")
#     return True


# for episode_name in os.listdir(input_dir):
#     if episode_name.startswith("."):
#         continue

#     episode_dir = input_dir.joinpath(episode_name)
#     for id in id_list:
#         rgb_dir = episode_dir.joinpath("rgb_" + str(id))
#         rgb_file_list = os.listdir(rgb_dir)
#         num_raw_images = len(rgb_file_list)
#         print(f"Checking for black images in {rgb_dir}")
#         with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
#             futures = set()
#             for i in range(len(rgb_file_list)):
#                 futures.add(
#                     executor.submit(
#                         check_black_images,
#                         rgb_file_list,
#                         rgb_dir,
#                         i,
#                         f"{episode_name} rgb_{id}",
#                     )
#                 )

#             completed, futures = concurrent.futures.wait(futures)
#             for f in completed:
#                 if not f.result():
#                     raise RuntimeError("Failed to read image!")
# exit()


# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

input_dir='/data/haoxiang/acp/flip_v3'
print("Reading data from input_dir: ", input_dir)
episode_names = os.listdir(input_dir)


def process_one_episode(root, episode_name, input_dir, id_list):
    if episode_name.startswith("."):
        return True

    # info about input
    # episode_id = episode_name[8:]
    # print(f"episode_name: {episode_name}, episode_id: {episode_id}")
    # episode_dir = input_dir.joinpath(episode_name)

    # acp的格式
    # episode_1727294514
    # 这个考虑修改上面那个路径读取逻辑
    # scene_0001

    episode_id = episode_name[6:]
    print(f"scene_name: {episode_name}, scene_id: {episode_id}")
    episode_dir = input_dir.joinpath(episode_name)

    # read rgb
    data_rgb = []
    data_rgb_time_stamps = []
    rgb_data_shapes = []
    for id in id_list:
        # rgb_dir = episode_dir.joinpath("rgb_" + str(id))
        rgb_dir = episode_dir.joinpath("cam_104122060902").joinpath("color")

        # 检查路径是否存在，防止报错
        if not rgb_dir.exists():
            print(f"Directory not found: {rgb_dir}")
            continue

        rgb_file_list = os.listdir(rgb_dir)

        rgb_file_list.sort()  # important!

        num_raw_images = len(rgb_file_list)

        img = cv2.imread(str(rgb_dir.joinpath(rgb_file_list[0])))

        rgb_data_shapes.append((num_raw_images, *img.shape))
        data_rgb.append(np.zeros(rgb_data_shapes[id], dtype=np.uint8))
        data_rgb_time_stamps.append(np.zeros(num_raw_images))

        print(f"Reading rgb data from: {rgb_dir}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = set()
            for i in range(len(rgb_file_list)):
                futures.add(
                    executor.submit(
                        image_read,
                        rgb_dir,
                        rgb_file_list,
                        i,
                        data_rgb[id],
                        data_rgb_time_stamps[id],
                    )
                )

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to read image!")

    # read low dim data
    data_ts_pose_fb = []
    data_robot_time_stamps = []
    data_wrench = []
    data_wrench_time_stamps = []
    print(f"Reading low dim data for : {episode_dir}")
    for id in id_list:
        json_path = episode_dir.joinpath("robot_pose_data_" + str(id) + ".json")
        df_robot_data = pd.read_json(json_path)
        data_robot_time_stamps.append(df_robot_data["robot_time_stamps"].to_numpy())
        data_ts_pose_fb.append(np.vstack(df_robot_data["ts_pose_fb"]))

        # read wrench data
        json_path = episode_dir.joinpath("wrench_data_" + str(id) + ".json")
        df_wrench_data = pd.read_json(json_path)
        data_wrench_time_stamps.append(df_wrench_data["wrench_time_stamps"].to_numpy())
        data_wrench.append(np.vstack(df_wrench_data["wrench"]))

    # get filtered force

    # 这个filtered不能基于data_wrench计算，需要基于原始的1000hz的数据全量算出来
    # 然后根据data_wrench里面的timestamp提取

    print(f"Computing filtered wrench for {episode_name}")

    # force_filtering_para = {
    #     "sampling_freq": 100,
    #     "cutoff_freq": 5,
    #     "order": 5,
    # }
    # ft_filter = LiveLPFilter(
    #     fs=500,
    #     # 这个采样频率需要基于真实数据修改
    #     cutoff=5,
    #     order=5,
    #     dim=6,
    # )
    # data_wrench_filtered = []
    # for id in id_list:
    #     data_wrench_filtered.append(np.array([ft_filter(y) for y in data_wrench[id]]))

    data_wrench_filtered = compute_aligned_filtered_wrench(
        episode_dir, 
        id_list, 
        data_wrench_time_stamps
    )

    # make time stamps start from zero
    # 不过既然我们的数据时间戳是统一的，其实也不用他这里这么麻烦
    time_offsets = []
    for id in id_list:
        time_offsets.append(data_rgb_time_stamps[id][0])
        time_offsets.append(data_robot_time_stamps[id][0])
        time_offsets.append(data_wrench_time_stamps[id][0])
    time_offset = np.min(time_offsets)
    for id in id_list:
        data_rgb_time_stamps[id] -= time_offset
        data_robot_time_stamps[id] -= time_offset
        data_wrench_time_stamps[id] -= time_offset

    # create output zarr
    print(f"Saving everything to : {output_dir}")
    recoder_buffer = EpisodeDataBuffer(
        store_path=output_dir,
        camera_ids=id_list,
        save_video=True,
        save_video_fps=60,
        data=root,
    )

    # save data using recoder_buffer
    rgb_data_buffer = {}
    for id in id_list:
        rgb_data = data_rgb[id]
        rgb_data_buffer.update({id: VideoData(rgb=rgb_data, camera_id=id)})
    recoder_buffer.create_zarr_groups_for_episode(rgb_data_shapes, id_list, episode_id)
    recoder_buffer.save_video_for_episode(
        visual_observations=rgb_data_buffer,
        visual_time_stamps=data_rgb_time_stamps,
        episode_id=episode_id,
    )
    recoder_buffer.save_low_dim_for_episode(
        ts_pose_command=data_ts_pose_fb,
        ts_pose_fb=data_ts_pose_fb,
        wrench=data_wrench,
        wrench_filtered=data_wrench_filtered,
        robot_time_stamps=data_robot_time_stamps,
        wrench_time_stamps=data_wrench_time_stamps,
        episode_id=episode_id,
    )
    return True


with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(
            process_one_episode,
            root,
            episode_name,
            input_dir,
            id_list,
        )
        for episode_name in episode_names
    ]
    for future in concurrent.futures.as_completed(futures):
        if not future.result():
            raise RuntimeError("Multi-processing failed!")

print("Finished reading. Now start generating metadata")
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()
buffer = zarr.open(output_dir)
meta = buffer.create_group("meta", overwrite=True)
episode_robot_len = []
episode_wrench_len = []
episode_rgb_len = []

# for id in id_list:
#     episode_robot_len.append([])
#     episode_wrench_len.append([])
#     episode_rgb_len.append([])

episode_robot_len = {id: [] for id in id_list}
episode_wrench_len = {id: [] for id in id_list}
episode_rgb_len = {id: [] for id in id_list}

count = 0
for key in buffer["data"].keys():
    episode = key
    ep_data = buffer["data"][episode]

    for id in id_list:
        # episode_robot_len[id].append(ep_data[f"ts_pose_fb_{id}"].shape[0])
        # episode_wrench_len[id].append(ep_data[f"wrench_{id}"].shape[0])
        # episode_rgb_len[id].append(ep_data[f"rgb_{id}"].shape[0])
        # print(
        #     f"Number {count}: {episode}: id = {id}: robot len: {episode_robot_len[id][-1]}, wrench_len: {episode_wrench_len[id][-1]} rgb len: {episode_rgb_len[id][-1]}"
        # )

        # --- 修改 : 使用字典 Key 访问 ---
        # 这里的 id 是 0
        r_len = ep_data[f"ts_pose_fb_{id}"].shape[0]
        w_len = ep_data[f"wrench_{id}"].shape[0]
        v_len = ep_data[f"rgb_{id}"].shape[0]

        episode_robot_len[id].append(r_len)
        episode_wrench_len[id].append(w_len)
        episode_rgb_len[id].append(v_len)
        
        print(f"Number {count}: {episode}: id={id}: robot={r_len}, wrench={w_len}, rgb={v_len}")

    count += 1

for id in id_list:
    meta[f"episode_robot{id}_len"] = zarr.array(episode_robot_len[id])
    meta[f"episode_wrench{id}_len"] = zarr.array(episode_wrench_len[id])
    meta[f"episode_rgb{id}_len"] = zarr.array(episode_rgb_len[id])

print(f"All done! Generated {count} episodes in {output_dir}")
print("The only thing left is to run postprocess_add_virtual_target_label.py")
