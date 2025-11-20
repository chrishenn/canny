from collections.abc import Generator
from pathlib import Path

import pytest
import torch as t
from canny.canny import Canny
from more_itertools import first, one
from torch import Tensor, allclose, device
from torch.export import export
from torch.fx import GraphModule
from torch.jit import RecursiveScriptModule
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage
from torchvision.utils import save_image


t.manual_seed(0)

datadir = Path(__file__).parent / "data"
golddir = Path(__file__).parent / "golden"


def tdata(imdir: Path) -> Tensor:
    tf = Compose([ToImage(), Resize([128, 128]), ToDtype(t.float, scale=True)])
    dataset = ImageFolder(imdir, tf)
    return first(one(DataLoader(dataset, batch_size=4)))


def gdata(imdir: Path) -> Tensor:
    tf = Compose([ToImage()])
    dataset = ImageFolder(imdir, tf)
    return first(one(DataLoader(dataset, batch_size=4)))


def tcases() -> Generator[tuple[Canny | RecursiveScriptModule | GraphModule, device, Tensor, Tensor]]:
    cannykwds = {"thresh_lo": 0.1, "thresh_hi": 0.2}

    data = tdata(datadir)
    gold = gdata(golddir)

    devs = [t.device("cpu"), t.device("cuda:0")]

    cannycls = Canny(**cannykwds)
    clss = [cannycls, t.jit.script(cannycls)]

    for dev in devs:
        for canny in clss:
            yield canny, dev, data, gold

    for dev in devs:
        canny = export(cannycls.to(dev), args=(data.to(dev),), kwargs={}).module()
        yield canny, dev, data, gold


@pytest.mark.parametrize(("canny", "dev", "data", "gold"), tcases())
def test_golden(canny: Canny | RecursiveScriptModule | GraphModule, dev: device, data: Tensor, gold: Tensor) -> None:
    canny = canny.to(dev)
    data = data.to(dev)
    gold = gold.to(dev)

    save = False

    for i, outp in enumerate(canny(data)):
        outp.sub_(outp.min())
        outp.div_(outp.max())

        if save:
            save_image(outp, golddir / f"{i}.png")
            continue

        # this is the scaling that save_image does
        outp_n = outp.mul(255).add_(0.5).clamp_(0, 255).to(t.uint8)
        assert allclose(gold[i], outp_n)


if __name__ == "__main__":
    pytest.main()
