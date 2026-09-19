"""
Time texture synthesis for every histogram matching mode and score the results, on whichever device is available.

python benchmark.py --sizes 256 512 1024 --modes sort cdf chol
"""

import argparse
from time import perf_counter

import torch

from evaluate import score
from optex import OptimalTexture
from util import load_styles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="style/graffiti.jpg")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--modes", type=str, nargs="+", default=["sort", "cdf", "chol"])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=1, help="Average over this many seeds.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"{'size':>6}{'mode':>6}{'seconds':>9}{'distance':>10}{'peak GB':>9}")

    with torch.no_grad():
        for size in args.sizes:
            styles = load_styles([args.style], size=size, scale=1, device=device)
            for mode in args.modes:
                texturizer = OptimalTexture(size=size, iters=args.iters, hist_mode=mode).to(device)
                seconds, distance = 0.0, 0.0
                for seed in range(args.seeds + 1):  # the first run only warms up
                    torch.manual_seed(seed)
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    t = perf_counter()
                    pastiche = texturizer(torch.rand(1, 3, size, size, device=device), styles)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    if seed > 0:
                        seconds += (perf_counter() - t) / args.seeds
                        distance += sum(score(pastiche.clamp(0, 1), styles[0], texturizer.encoder)) / 5 / args.seeds
                peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else float("nan")
                print(f"{size:>6}{mode:>6}{seconds:>9.1f}{distance:>10.4f}{peak:>9.2f}")
