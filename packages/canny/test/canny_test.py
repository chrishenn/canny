from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch as t
import torchvision as tv
from more_itertools import one
from torch import Tensor
from torch.export import export
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tf

from canny.canny import Canny


def tensor_imshow(_img: Tensor, dpi: int = 100, axis: str = "off") -> None:
    _img = _img.cpu()
    _img = _img.sub(_img.min())
    _img = _img.div(_img.max()).clamp(0, 1)

    plt.figure(dpi=dpi)
    plt.axis(axis)
    plt.imshow(tf.ToPILImage()(_img))
    plt.show(block=False)


def test_simple_visual() -> None:
    folder = Path(__file__).parent / "data"

    dev = t.device("cuda:0")
    # dev = t.device('cpu')

    resize = tf.Resize(128)
    normalize = tf.Normalize(mean=0.0, std=0.1)
    dataset = tv.datasets.ImageFolder(folder, tf.Compose([resize, tf.ToTensor(), normalize]))
    data, _ = one(DataLoader(dataset, batch_size=4))
    data = data.to(dev)

    # python class
    canny = Canny(thresh_lo=0.1, thresh_hi=0.2).to(dev)

    # torchscript-jit, or
    # canny = t.jit.script(canny)

    # newer, traced torch-export
    canny = export(canny, args=(data,), kwargs={}).module()

    for img in data:
        tensor_imshow(img, dpi=100)
    for img in canny(data):
        tensor_imshow(img, dpi=100)
    plt.show(block=True)


if __name__ == "__main__":
    pytest.main()
