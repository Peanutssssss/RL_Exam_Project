import argparse, os, glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

def load_scalar(run_dir, tag):
    # Fetch the latest event file
    events = sorted(glob.glob(os.path.join(run_dir, "**", "events.*"), recursive=True))
    if not events:
        return pd.Series(dtype=float)
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return pd.Series(dtype=float)
    scal = ea.Scalars(tag)
    df = pd.DataFrame([(s.step, s.value) for s in scal], columns=["step", os.path.basename(run_dir)])
    return df.set_index("step")[os.path.basename(run_dir)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="The parent directory, which contains the individual run subdirectories")
    ap.add_argument("--runs", nargs="+", required=True, help="The name of the run directory to be merged, e.g. PPO_1 PPO_2")
    ap.add_argument("--tag", required=True, help="e.g. eval/mean_reward")
    ap.add_argument("--out_run", default="avg_run", help="The name of the new run directory to write back to")
    args = ap.parse_args()

    series = []
    for r in args.runs:
        s = load_scalar(os.path.join(args.logdir, r), args.tag)
        if not s.empty:
            series.append(s)

    if not series:
        print("No scalar data was read")
        return

    # outer join, aligned by step; can be changed to .interpolate() for linear interpolation
    df = pd.concat(series, axis=1).sort_index()
    mean = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, ddof=0, skipna=True)

    # Writing back a new TensorBoard log
    out = os.path.join(args.logdir, args.out_run)
    os.makedirs(out, exist_ok=True)
    writer = SummaryWriter(out)
    for step, val in mean.items():
        writer.add_scalar(args.tag, float(val), global_step=int(step))
    # Optional: write std also as an extra tag
    for step, val in std.items():
        writer.add_scalar(args.tag + "_std", float(val), global_step=int(step))
    writer.close()
    print(f"done: {out}")

if __name__ == "__main__":
    main()
