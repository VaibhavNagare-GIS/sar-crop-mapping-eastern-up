"""Array preparation for the neural-network experiments (numpy only, so it can be tested without PyTorch).
Chip stacks hold 80 bands: ascending VV (20 windows), VH (20), then descending VV, VH, in dB, as written by p5_chip_features.py."""
import numpy as np

NO = -32768   # no-data value of the int16 stacks


def dequantise(q):
    """int16 stack (dB x 100) -> float32 dB with NaN for no data. Same as p5_chip_features.dequantise."""
    return np.where(q == NO, np.nan, q / 100.0).astype(np.float32)


def fill_windows(a):
    """a: (T, ...) with NaN. Windows with no value anywhere on the chip are interpolated linearly in time (nearest window at the ends)."""
    T = a.shape[0]
    valid = np.isfinite(a.reshape(T, -1)).any(axis=1)
    if not valid.any():
        raise ValueError("no valid window")
    idx, out = np.flatnonzero(valid), a.copy()
    for t in np.flatnonzero(~valid):
        lo, hi = idx[idx < t], idx[idx > t]
        if len(lo) and len(hi):
            w = (t - lo[-1]) / (hi[0] - lo[-1])
            out[t] = (1 - w) * a[lo[-1]] + w * a[hi[0]]
        else:
            out[t] = a[lo[-1]] if len(lo) else a[hi[0]]
    return out


def chip_array(stack):
    """(80, h, w) dB stack -> (6, 20, h, w): per orbit VV, VH and VH minus VV, gaps filled."""
    ch = []
    for s in range(stack.shape[0] // 40):
        vv, vh = fill_windows(stack[s * 40:s * 40 + 20]), fill_windows(stack[s * 40 + 20:s * 40 + 40])
        ch += [vv, vh, vh - vv]
    return np.stack(ch).astype(np.float32)


def channel_stats(arrays):
    """Mean and standard deviation per channel over time and pixels, ignoring NaN."""
    s = np.stack([[np.nanmean(a[c]) for c in range(a.shape[0])] for a in arrays]).mean(0)
    v = np.stack([[np.nanmean((a[c] - s[c]) ** 2) for c in range(a.shape[0])] for a in arrays]).mean(0)
    return s.astype(np.float32), np.sqrt(v).astype(np.float32) + 1e-6


def normalise(a, mean, std):
    return np.nan_to_num((a - mean[:, None, None, None]) / std[:, None, None, None]).astype(np.float32)


def label_image(rows, cols, yi, h, w):
    img = np.full((h, w), -1, dtype=np.int64)
    img[rows, cols] = yi
    return img


def chip_blocks(chip_of, block_of):
    """Chips (sorted) and the 0.1 degree block that most of their labelled pixels belong to. Shared with p6_densecv.py so the folds agree."""
    chips = np.array(sorted(set(chip_of)))
    blocks = []
    for c in chips:
        vals, cnt = np.unique(np.asarray(block_of)[chip_of == c], return_counts=True)
        blocks.append(vals[cnt.argmax()])
    return chips, np.array(blocks)
