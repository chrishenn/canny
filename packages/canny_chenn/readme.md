<h1 align="center" style="margin-top: 0px;">Canny Edges</h1>
<div id="img0" align="center">
    <img src="https://raw.githubusercontent.com/chrishenn/canny/refs/heads/main/packages/canny_chenn/doc/images/img1.png" width=200 alt="img1_in">
    <img src="https://raw.githubusercontent.com/chrishenn/canny/refs/heads/main/packages/canny_chenn/doc/images/img2.png" width=200 alt="img2_in">
    <img src="https://raw.githubusercontent.com/chrishenn/canny/refs/heads/main/packages/canny_chenn/doc/images/img3.png" width=200 alt="img3_in">
</div>
<div id="img0" align="center">
    <img src="https://raw.githubusercontent.com/chrishenn/canny/refs/heads/main/packages/canny_chenn/doc/images/img1_out.png" width=200 alt="img1_out">
    <img src="https://raw.githubusercontent.com/chrishenn/canny/refs/heads/main/packages/canny_chenn/doc/images/img2_out.png" width=200 alt="img2_out">
    <img src="https://raw.githubusercontent.com/chrishenn/canny/refs/heads/main/packages/canny_chenn/doc/images/img3_out.png" width=200 alt="img3_out">
</div>

&emsp;

A simple `Torch.nn.Module` to return an image mask representing edges found by the Canny Edge-Finding algorithm.


---

# Usage

Supports:

- Linux only
- PyTorch Tensor images formatted in image batches [B, C, h, w], with float32 data, and values spanning [0,1]
- Any number of channels C
- TorchScript jit script, as well as the newer, traced torch-export, on the Canny `nn.Module` class

pip install

```bash
pip install canny-chenn
# or,
pixi add --pypi canny_chenn
```

then

```python
from torch import Tensor
from canny_chenn import Canny
canny = Canny()
img_batch = Tensor([[[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]])
edge_mask = canny(img_batch)
# tensor([[[[0.0000, 0.6333, 0.0000, 0.0000],
#           [0.0000, 0.6333, 0.0000, 0.0000],
#           [0.0000, 0.6333, 0.0000, 0.0000]]]])
```

---

# Dev

I use mise to handle project tools. Install it if you don't have it

```bash
# clone, install tools
git clone https://github.com/chrishenn/canny.git
cd canny
mise i

# list available just recipes
just --list

Available recipes:
    interactive # Detect and display canny edges on a set of test images. Use ctrl+c to kill
    lint        # run project formatters/linters [alias: l]
    test        # run basic tests
    unsafe      # run ruff with unsafe-fixes
    
# build distribution pkgs locally
uv build --package canny_chenn

# push tag to trigger pypi to build pkgs and publish to pypi
git tag -a v1.0.3 -m v1.0.3 -f && git push --tags -f
```
