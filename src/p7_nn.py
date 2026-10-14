"""Phase 7: TempCNN (pixel time series) and a small U-Net (patches with spatial context) on the dense chips. Run on the GPU machine.
python src/p7_nn.py --model tempcnn|unet [--smoke] [--max-chips N] [--epochs E] [--iters I]
Same labelled pixels, chip-level folds and scoring as p6_densecv.py, so the numbers are directly comparable with the Random Forest rows
(pixel 0.584 and 5 x 5 mean filter 0.609 on the full run). Rule: adopt only if macro-F1 after the 5 x 5 filter rises by 0.02 or more
over the Random Forest and AUC does not fall. --smoke runs 2 folds on 24 chips with a few steps, to check the code and time it."""
import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from sklearn.model_selection import GroupKFold

import nnprep
import p3_sweep as sw
import p4_final as p4

CLASSES = np.array([1, 2, 6])


class TempCNN(nn.Module):
    """1D convolutions over the 20 windows of one pixel (Pelletier et al. 2019 style)."""
    def __init__(self, c_in, t, n_cls, w=64, drop=0.3):
        super().__init__()
        layers, c = [], c_in
        for _ in range(3):
            layers += [nn.Conv1d(c, w, 5, padding=2), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(drop)]
            c = w
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(w * t, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop), nn.Linear(256, n_cls))

    def forward(self, x):          # (B, C, T)
        return self.net(x)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Small U-Net; input is the whole time series stacked as channels (6 x 20 = 120). Height and width must be divisible by 8."""
    def __init__(self, c_in, n_cls, w=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = block(c_in, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2), block(8 * w, 4 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2), block(4 * w, 2 * w)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2), block(2 * w, w)
        self.out = nn.Conv2d(w, n_cls, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def class_weights(y, device):
    cnt = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    return torch.tensor(len(y) / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32, device=device)


def pixels_of(arr, c, rows, cols):
    return arr[c][:, :, rows, cols].transpose(2, 0, 1).astype(np.float32)          # (n, 6, T)


def train_tempcnn(Xtr, ytr, Xva, yva, epochs, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TempCNN(Xtr.shape[1], Xtr.shape[2], len(CLASSES)).to(device)
    w = class_weights(ytr, device)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    best, state = float("inf"), None
    Xv, yv = torch.from_numpy(Xva).to(device), torch.from_numpy(yva).to(device)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            if len(b) < 2:
                continue
            loss = F.cross_entropy(model(torch.from_numpy(Xtr[b]).to(device)), torch.from_numpy(ytr[b]).to(device), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = F.cross_entropy(model(Xv), yv, weight=w).item()
        if v < best:
            best, state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval()


def predict_tempcnn(model, a, device, step=8192):
    c, t, h, w = a.shape
    X = a.reshape(c, t, h * w).transpose(2, 0, 1).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), step):
            out.append(F.softmax(model(torch.from_numpy(X[i:i + step]).to(device)), 1).cpu().numpy())
    return np.concatenate(out).reshape(h, w, -1)


def sample_batch(arr, lab, chips, size, batch, rng):
    xs, ys = [], []
    while len(xs) < batch:
        c = chips[rng.integers(len(chips))]
        a, l = arr[c], lab[c]
        h, w = l.shape
        for _ in range(10):
            r, q = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
            if (l[r:r + size, q:q + size] >= 0).sum() >= 8:
                break
        else:
            continue
        x, y = a[:, :, r:r + size, q:q + size].reshape(-1, size, size), l[r:r + size, q:q + size]
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
    return torch.from_numpy(np.stack(xs).astype(np.float32)), torch.from_numpy(np.stack(ys))


def predict_unet(model, a, device):
    c, t, h, w = a.shape
    with torch.no_grad():
        x = torch.from_numpy(a.reshape(c * t, h, w)[None].astype(np.float32)).to(device)
        return F.softmax(model(x), 1)[0].permute(1, 2, 0).cpu().numpy()


def train_unet(arr, lab, tr, va, iters, device, seed, size=64, batch=16, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    c, t = arr[tr[0]].shape[:2]
    model = UNet(c * t, len(CLASSES)).to(device)
    y_all = np.concatenate([lab[ch][lab[ch] >= 0] for ch in tr])
    w = class_weights(y_all, device)
    amp = device.type == "cuda"
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    best, state, every = float("inf"), None, max(1, iters // 6)
    for it in range(1, iters + 1):
        model.train()
        x, y = sample_batch(arr, lab, tr, size, batch, rng)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x.to(device)), y.to(device), weight=w, ignore_index=-1)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if it % every == 0 or it == iters:
            model.eval()
            with torch.no_grad():
                v = float(np.mean([F.cross_entropy(model(torch.from_numpy(arr[ch].reshape(-1, *arr[ch].shape[2:])[None].astype(np.float32)).to(device)),
                                                   torch.from_numpy(lab[ch][None]).to(device), weight=w, ignore_index=-1).item() for ch in va]))
            if v < best:
                best, state = v, {k: x_.detach().clone() for k, x_ in model.state_dict().items()}
            if log:
                log(f"  iteration {it}/{iters}: train loss {loss.item():.3f}, validation loss {v:.3f}")
    model.load_state_dict(state)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tempcnn", "unet"], required=True)
    ap.add_argument("--asc", default="dense_asc_mt2")
    ap.add_argument("--desc", default="dense_desc_mt2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--max-chips", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    if a.smoke:
        a.max_chips, a.epochs, a.iters = 24, 2, 60
    data = Path(a.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    parts, meta = sw.load("+".join(str(data / "phase2" / n) for n in (a.asc, a.desc)))
    pix = pd.read_csv(data / "phase1" / "pixels.csv", dtype={"chip": str})
    if a.max_chips:
        keep_chips = np.random.default_rng(0).permutation(pix.loc[meta.index].chip.unique())[:a.max_chips]
        keep = pix.loc[meta.index].chip.isin(keep_chips).to_numpy()
        parts, meta = [p[keep] for p in parts], meta[keep]
    parts, meta = p4.prep(parts, meta, "3")
    px = pix.loc[meta.index]
    yi = np.searchsorted(CLASSES, meta.label.to_numpy())
    chip_of, rows, cols = px.chip.to_numpy(), px.row.to_numpy(), px.col.to_numpy()
    chips, block = nnprep.chip_blocks(chip_of, px.block.to_numpy())
    rows_of = {c: np.flatnonzero(chip_of == c) for c in chips}

    t0 = time.time()
    raw = {c: nnprep.chip_array(np.concatenate([nnprep.dequantise(np.load(data / "phase5" / n / f"{c}.npy")) for n in (a.asc, a.desc)])) for c in chips}
    mean, std = nnprep.channel_stats([raw[c] for c in chips[:40]])
    arr = {c: nnprep.normalise(raw[c], mean, std).astype(np.float16) for c in chips}
    del raw
    h, w = arr[chips[0]].shape[2:]
    lab = {c: nnprep.label_image(rows[rows_of[c]], cols[rows_of[c]], yi[rows_of[c]], h, w) for c in chips}
    print(f"{len(chips)} chips prepared in {time.time() - t0:.0f}s")

    n_splits = min(2 if a.smoke else 5, len(set(block)))
    P = {"raw": np.zeros((len(yi), len(CLASSES))), "smooth5": np.zeros((len(yi), len(CLASSES)))}
    rng = np.random.default_rng(a.seed)
    t1 = time.time()
    for fold, (tr_i, te_i) in enumerate(GroupKFold(n_splits=n_splits).split(chips, groups=block), 1):
        trc, tec = list(chips[tr_i]), list(chips[te_i])
        rng.shuffle(trc)
        nva = max(1, len(trc) // 10)
        va, tr = trc[:nva], trc[nva:]
        if a.model == "tempcnn":
            def xy(cs):
                return (np.concatenate([pixels_of(arr, c, rows[rows_of[c]], cols[rows_of[c]]) for c in cs]), np.concatenate([yi[rows_of[c]] for c in cs]))
            (Xt, yt), (Xv, yv) = xy(tr), xy(va)
            model = train_tempcnn(Xt, yt, Xv, yv, a.epochs, device, a.seed)
            predict = lambda c: predict_tempcnn(model, arr[c], device)
        else:
            model = train_unet(arr, lab, tr, va, a.iters, device, a.seed, log=print)
            predict = lambda c: predict_unet(model, arr[c], device)
        for c in tec:
            pr = predict(c)
            idx = rows_of[c]
            P["raw"][idx] = pr[rows[idx], cols[idx]]
            P["smooth5"][idx] = uniform_filter(pr, size=(5, 5, 1), mode="nearest")[rows[idx], cols[idx]]
        print(f"fold {fold}/{n_splits} done, {time.time() - t1:.0f}s")
    out = []
    for k, v in P.items():
        r = sw.score(v, yi)
        print(f"{a.model} {k:8s} F1 {r['f1']:.3f} bal {r['bal']:.3f} OA {r['oa']:.3f} AUC {r['auc']:.3f}")
        out.append(dict(run=dt.datetime.now().isoformat(timespec="seconds"), model=a.model, variant=k, chips=len(chips), folds=n_splits,
                        epochs=a.epochs, iters=a.iters, smoke=a.smoke, f1=r["f1"], bal=r["bal"], oa=r["oa"], auc=r["auc"]))
    if a.smoke:
        print(f"smoke run finished in {time.time() - t0:.0f}s on {len(chips)} chips: the code runs end to end")
        return
    path = data / "phase7" / "nn.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    pd.concat([pd.read_csv(path), df]).to_csv(path, index=False) if path.exists() else df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
