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

Supports:

- PyTorch Tensor images formatted in image batches [B, C, h, w], with float32 data, and values spanning [0,1]
- any number of channels C
- TorchScript jit script, as well as the newer, traced torch-export, on the Canny `nn.Module` class

---

# Usage

I use mise to handle project tools. Install it if you don't have it

```bash
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
```
