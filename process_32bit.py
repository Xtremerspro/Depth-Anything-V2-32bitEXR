import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2


def run_processing(input_path, output_path, encoder="vitl"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    model = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f"checkpoints/depth_anything_v2_{encoder}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(DEVICE).eval()

    # Get files
    if os.path.isfile(input_path):
        files = [input_path]
        outdir = output_path
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    elif os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*"))
        outdir = output_path
        os.makedirs(outdir, exist_ok=True)
    else:
        print(f"Input {input_path} not found.")
        return

    print(f"Processing {len(files)} files into 32-bit EXR...")

    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
            print(f"Processing: {os.path.basename(file)}")

            raw_img = cv2.imread(file)

            # Inference (Result is a float32 array)
            depth = model.infer_image(raw_img)

            # --- 32-BIT FLOAT NORMALIZATION ---
            # We normalize to 0.0 - 1.0 range, but keep it as float32.
            # This ensures Blender understands the scale, but retains infinite precision.
            depth_min = depth.min()
            depth_max = depth.max()

            # Avoid division by zero if image is flat
            if depth_max - depth_min > 0:
                depth_32bit = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_32bit = depth

            # Save as .exr (OpenCV automatically uses float32 for EXR)
            filename = os.path.basename(file)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(outdir, f"{name}.exr")

            cv2.imwrite(save_path, depth_32bit)

    print("Done! EXR files saved to:", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-path",
        type=str,
        default="./input_images",
        help="Path to input image or folder",
    )
    parser.add_argument(
        "--outdir", type=str, default="./output_32bit", help="Output folder"
    )

    args = parser.parse_args()
    run_processing(args.img_path, args.outdir)
