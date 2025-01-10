# process_data.py
import asyncio
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import shutil
import os
from tqdm.asyncio import tqdm_asyncio
import time

IMAGE_EXTENSION = ['.jpg', '.jpeg', '.png']


def timer_decorator(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to execute")
        return result

    return wrapper


def make_image_height_width_same(image: np.ndarray, target_size: int = None) -> np.ndarray:
    height, width = image.shape[:2]
    size = max(height, width)
    square_img = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = image

    if target_size is None:
        target_size = 2 ** (size - 1).bit_length()

    if size != target_size:
        square_img = cv2.resize(square_img, (target_size, target_size),
                                interpolation=cv2.INTER_LANCZOS4)

    return square_img


def process_single_image(image_path, output_dir, target_size):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = make_image_height_width_same(image, target_size)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), processed_image)


@timer_decorator
async def process_data(data_dir: Path, output_dir: Path, target_size: int = 512):
    if not (target_size & (target_size - 1) == 0):
        raise ValueError("target_size must be a power of 2")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_json_path = data_dir / 'dataset.json'
    target_json_path = output_dir / 'dataset.json'
    shutil.copy(source_json_path, target_json_path)

    image_files = []
    for ext in IMAGE_EXTENSION:
        image_files.extend(data_dir.glob(f'*{ext}'))

    loop = asyncio.get_event_loop()
    workers = max(os.cpu_count(), 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        tasks = []
        for image_path in image_files:
            task = loop.run_in_executor(executor, process_single_image, image_path, output_dir, target_size)
            tasks.append(task)
        await tqdm_asyncio.gather(*tasks, desc="Processing images")


async def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--target_size', type=int, default=512,
                        help='Target size for output images (must be power of 2)')
    args = parser.parse_args()
    await process_data(args.data_dir, args.output_dir, args.target_size)


if __name__ == '__main__':
    asyncio.run(main())