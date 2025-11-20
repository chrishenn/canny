from pathlib import Path

import matplotlib.pyplot as plt
import torch as t
import torchvision as tv
from canny.canny import Canny
from more_itertools import one
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tf
from torchvision.transforms.v2.functional import to_pil_image


def tensor_imshow(_img: Tensor, dpi: int = 100, axis: str = "off") -> None:
    _img = _img.cpu()
    _img = _img.sub(_img.min())
    _img = _img.div(_img.max())

    plt.figure(dpi=dpi)
    plt.axis(axis)
    plt.imshow(to_pil_image(_img))
    plt.show(block=False)


def simple_visual() -> None:
    datadir = Path(__file__).parent / "data"

    dev = t.device("cpu")

    resize = tf.Resize(128)
    normalize = tf.Normalize(mean=0.0, std=0.1)
    dataset = tv.datasets.ImageFolder(datadir, tf.Compose([resize, tf.ToTensor(), normalize]))
    data, _ = one(DataLoader(dataset, batch_size=4))
    data = data.to(dev)

    canny = Canny(thresh_lo=0.1, thresh_hi=0.2).to(dev)

    for img in data:
        tensor_imshow(img, dpi=100)
    for img in canny(data):
        tensor_imshow(img, dpi=100)
    plt.show(block=True)


if __name__ == "__main__":
    simple_visual()
