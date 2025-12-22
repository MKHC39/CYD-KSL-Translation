import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from Dataset.cache_clips import CACHE_DIR
from pydata import KSLStemDataset, collate_pad_time
from modules.resnet import FrozenResNetMeanClassifier
from cached_dataset import KSLCachedDataset, collate_pad_time


def train_one_epoch(model, loader, optimiser, device):
    model.train()
    crit = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    for clips, ys, lengths, stems in loader:
        clips = clips.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        logits = model(clips, lengths)
        loss = crit(logits, ys)
        loss.backward()
        optimiser.step()

        bs = ys.size(0)
        total_loss += float(loss.item()) * bs
        total += bs
        correct += (logits.argmax(dim=1) == ys).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    for clips, ys, lengths, stems in loader:
        clips = clips.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        logits = model(clips, lengths)
        loss = crit(logits, ys)

        bs = ys.size(0)
        total_loss += float(loss.item()) * bs
        total += bs
        correct += (logits.argmax(dim=1) == ys).sum().item()

    return total_loss / total, correct / total


def main():
    # -------- EDIT THESE --------
    W_START = 1501
    W_END = 1520          # start small
    ANGLES = ["D", "F", "L", "R", "U"]

    BATCH_SIZE = 4
    EPOCHS = 100
    LR = 1e-3
    VAL_FRAC = 0.2
    # ----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device chosen:", device)

    ds = KSLCachedDataset(cache_dir=CACHE_DIR, w_start=W_START, w_end=W_END, angles=ANGLES)

    num_classes = (W_END - W_START + 1)
    model = FrozenResNetMeanClassifier(num_classes=num_classes, pretrained=True).to(device)

    print("model device:", next(model.parameters()).device)

    n_val = int(len(ds) * VAL_FRAC)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # try 2/4/8 depending on CPU
        collate_fn=collate_pad_time,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_pad_time,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    optimiser = torch.optim.Adam(model.classifier.parameters(), lr=LR)  # only train classifier

    clips, ys, lengths, *_ = next(iter(train_loader))
    print("before move:", clips.device, ys.device)
    clips = clips.to(device, non_blocking=True)
    ys = ys.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)
    print("after move:", clips.device, ys.device, lengths.device)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimiser, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, device)
        print(f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")


if __name__ == "__main__":
    main()
