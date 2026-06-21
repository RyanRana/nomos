"""Assemble captured PNG frame sequences into looping GIFs."""
import glob
import os

import numpy as np
import imageio.v2 as imageio
from PIL import Image

SRC = "/tmp/gif_frames"
OUT = "runs/artifacts"
os.makedirs(OUT, exist_ok=True)
W = 820  # downscale width (keeps GIF size sane vs 1280px screenshots)

for view in sorted(os.listdir(SRC)):
    files = sorted(glob.glob(f"{SRC}/{view}/frame*.png"))
    if not files:
        continue
    frames = []
    for f in files:
        im = Image.open(f).convert("RGB")
        h = int(im.height * W / im.width)
        frames.append(np.asarray(im.resize((W, h), Image.LANCZOS)))
    dst = f"{OUT}/champion_{view}.gif"
    imageio.mimsave(dst, frames, format="GIF", duration=0.085, loop=0)
    mb = os.path.getsize(dst) / 1e6
    print(f"{dst}  ({len(frames)} frames, {mb:.1f} MB)")
