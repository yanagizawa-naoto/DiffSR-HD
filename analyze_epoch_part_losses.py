# analyze_epoch_part_losses.py
# This script reads a `train.log` file and computes summary statistics (count, avg, min, max)
# of the reported loss for each epoch and each part.

import re
import sys
from collections import defaultdict

def parse_train_log(log_path):
    """
    Parses the train.log file and extracts losses per (epoch, part).
    Returns a dict mapping (epoch, part) -> list of losses.
    """
    ep_part_losses = defaultdict(list)
    # Matches: "Epoch 1 Part 2 Batch 123 Loss 0.045123"
    pattern = re.compile(r"Epoch\s+(\d+)\s+Part\s+(\d+)\s+Batch\s+\d+\s+Loss\s+([0-9.]+)")
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                part  = int(m.group(2))
                loss  = float(m.group(3))
                ep_part_losses[(epoch, part)].append(loss)
    return ep_part_losses

def compute_and_print(ep_part_losses):
    """
    Prints summary table of count, average, min, max loss per epoch and part.
    """
    print(f"{'Epoch':<6} {'Part':<6} {'Count':<6} {'Avg Loss':<10} {'Min Loss':<10} {'Max Loss':<10}")
    print("-" * 60)
    for (epoch, part) in sorted(ep_part_losses):
        losses = ep_part_losses[(epoch, part)]
        count = len(losses)
        avg   = sum(losses) / count if count else float('nan')
        mn    = min(losses) if losses else float('nan')
        mx    = max(losses) if losses else float('nan')
        print(f"{epoch:<6} {part:<6} {count:<6} {avg:<10.6f} {mn:<10.6f} {mx:<10.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_epoch_part_losses.py <path/to/train.log>")
        sys.exit(1)
    log_path = sys.argv[1]
    ep_part_losses = parse_train_log(log_path)
    if not ep_part_losses:
        print("No 'Epoch Part' loss entries found in the log.")
    else:
        compute_and_print(ep_part_losses)

# Save this as analyze_epoch_part_losses.py and run:
#   python analyze_epoch_part_losses.py train.log
