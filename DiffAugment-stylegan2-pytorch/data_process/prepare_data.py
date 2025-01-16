# prepare_data.py
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import shutil
import json
import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to execute")
        return result

    return wrapper


IMAGE_EXTENSION = [".jpg", ".jpeg", ".png"]

"""
处理数据：
1. 把所有的图片复制过来(需要重命名一下) 
2. 然后生成json数据
"""


def copy_sub_dir(source_dir: Path, target_dir: Path):
    sub_dir_name = source_dir.name
    for exten in IMAGE_EXTENSION:
        for idx, img_path in enumerate(source_dir.glob(f"*{exten}")):
            img_name = img_path.name
            target_path = target_dir / f"{sub_dir_name}_{img_name}"
            shutil.copy(img_path, target_path)


async def copy_images(source_dir: Path, target_dir: Path):
    """
    explicit for loop: copy_images took 73.3418619632721 seconds to execute
    async gather: 18.989242792129517
    """
    sub_dirs = list(source_dir.iterdir())
    sub_dirs = [sub_dir for sub_dir in sub_dirs if sub_dir.is_dir()]
    #
    # for sub_dir in tqdm(sub_dirs, desc="Copying images", unit="directory"):
    #     if sub_dir.is_dir():
    #         copy_sub_dir(sub_dir, target_dir)
    loop = asyncio.get_event_loop()
    workers = max(os.cpu_count(), 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        tasks = []
        for sub_dir in sub_dirs:
            task = loop.run_in_executor(executor, copy_sub_dir, sub_dir, target_dir)
            tasks.append(task)
        await tqdm_asyncio.gather(*tasks)


def make_json_data(dir_path: Path):
    json_path = dir_path / "dataset.json"
    files = []
    for file in dir_path.iterdir():
        if file.is_file():
            file_name = file.name
            files.append(file_name)
    labels = {k: 0 for k in files}
    to_save_dict = {"labels": labels}
    with open(json_path, "w") as f:
        json.dump(to_save_dict, f, indent=4)


@timer_decorator
def prepared_data(source_dir: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(copy_images(source_dir, target_dir))
    # copy_images(source_dir, target_dir)
    make_json_data(target_dir)


def main():
    # parser = ArgumentParser()
    # parser.add_argument('--data_dir', type=Path, required=True)
    # parser.add_argument('--output_dir', type=Path, required=True)
    #
    # args = parser.parse_args()
    data_dir = r"C:\Users\23174\Desktop\GitHub Project\SVD-deposition-etching\data"
    output_dir = r"C:\Users\23174\Desktop\GitHub Project\data-efficient-gans-baseline\data\deposition_data"
    prepared_data(Path(data_dir), Path(output_dir))


if __name__ == "__main__":
    main()
