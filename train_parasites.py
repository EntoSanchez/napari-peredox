import logging
from pathlib import Path

from napari_peredox._stardist import train_stardist

logging.basicConfig(level=logging.INFO, format="[cellpose] %(message)s")


def log(msg):
    print(msg, flush=True)


out = train_stardist(
    training_dir=Path("annotations/training_data"),
    model_dir=Path("annotations/stardist_model"),
    mode="parasites",
    seg_channel=0,
    n_epochs=100,
    progress_cb=log,
)
print(f"Done — model saved to {out}", flush=True)
