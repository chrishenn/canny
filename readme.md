<h1 align="center" style="margin-top: 0px;">Canny Edges</h1>
<div id="img0" align="center">
    <img src="doc/images/img1.png" width=300 alt="img1_in">
    <img src="doc/images/img1_out.png" width=300 alt="img1_out">
</div>
<div id="img0" align="center">
    <img src="doc/images/img2.png" width=300 alt="img2_in">
    <img src="doc/images/img2_out.png" width=300 alt="img2_out">
</div>
<div id="img1" align="center">
    <img src="doc/images/img3.png" width=300 alt="img3_in">
    <img src="doc/images/img3_out.png" width=300 alt="img3_out">
</div>

&emsp;

A simple `Torch.nn.Module` to return an image mask representing edges found by the Canny Edge-Finding algorithm.

Supports:

- PyTorch Tensor images formatted in image batches [B, C, h, w], with float32 data, and values spanning [0,1]
- any number of channels C
- TorchScript jit script, as well as the newer, traced torch-export, on the Canny `nn.Module` class

---

# Usage

```bash
# clone
git clone https://github.com/chrishenn/canny.git
cd canny

# bash: install mise
curl https://mise.run | sh

# activate mise in project dir, install tools
mise activate
mise trust
mise install

# list available just recipes
just -l

  Available recipes:
    init        # run once to initialize env
    lint        # run project formatters/linters [alias: l]
    ruff_unsafe # run ruff unsafe-fixes
    test        # Detect and display canny edges on a set of test images. Use ctrl+c to kill

# run canny edge detector on test files and display the resulting edge mask - kill with ctrl+c
just test
```
