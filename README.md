---
title: NST
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.14.0
python_version: '3.11'
app_file: app.py
pinned: false
license: unknown
short_description: AdaIN neural style transfer for uploaded images
---

# AdaIN Neural Style Transfer

Repaint any photo in the style of any artwork in a single forward pass. This Gradio app uses **Adaptive Instance Normalization (AdaIN)**, from Huang & Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017). It pairs a frozen VGG-19 encoder with a trained decoder, so it works on styles it has never seen without any per-style training.

**Live demo:** [huggingface.co/spaces/Amanprime/NST](https://huggingface.co/spaces/Amanprime/NST)

---

## Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Using the App](#using-the-app)
- [Deploying Your Own Space](#deploying-your-own-space)

---

## Features

- **Arbitrary styles:** upload any content image and any style image.
- **Style strength slider** (0 to 1) to blend between the original content and the full style.
- **Two ready-made examples:** Lenna with a pencil-sketch style, and the Golden Gate Bridge in the style of *La Muse*.
- **Handles real photos:** EXIF rotation is respected, and images are scaled down to at most 512 px on the longest side, keeping their aspect ratio.
- **Runs anywhere:** uses a GPU when one is available and falls back to the CPU otherwise.
- **Request queue** (up to 8 waiting jobs), so several users can share the Space.

---

## How It Works

### 1. Preprocessing

Both images are rotated according to their EXIF data, converted to RGB, shrunk with a Lanczos filter so the longest side is at most **512 px**, and turned into tensors with values in [0, 1].

### 2. Encoding

The **VGG-19 encoder** (`VGGEncoder` in `utils/models.py`) is loaded from `weights/vgg_normalised.pth` and cut off after the `relu4_1` layer. It is split into four blocks ending at `relu1_1`, `relu2_1`, `relu3_1` and `relu4_1`, and all its weights are frozen.

Its first layer is a fixed 1×1 convolution that converts the RGB input into the colour format the original VGG weights expect. At inference time only the `relu4_1` output is used: **512 feature channels at 1/8 of the image resolution**.

### 3. Adaptive Instance Normalization

AdaIN keeps the *layout* of the content features and gives them the *statistics* of the style features. For every channel it takes the mean μ and standard deviation σ over the spatial dimensions:

```
AdaIN(x, y) = σ(y) · ( (x − μ(x)) / σ(x) ) + μ(y)
```

Here `x` is the content feature map and `y` is the style feature map. A small ε (1e-5) is added to the variance to avoid dividing by zero. This is `adaptive_instance_normalization()` and `calc_mean_std()` in `utils/utils.py`.

### 4. Style strength

The slider value α blends the stylised features with the original content features before decoding:

```
t = α · AdaIN(f_content, f_style) + (1 − α) · f_content
```

- **α = 1** gives the full style.
- **α = 0** reconstructs the content image.

### 5. Decoding

The **decoder** (`Decoder` in `utils/models.py`, weights in `weights/decoder_final.pth`) mirrors the encoder:

- reflection-padded 3×3 convolutions with ReLU
- nearest-neighbour upsampling ×2, three times
- channels going 512 → 256 → 128 → 64 → 3

The result is clamped to [0, 1] and returned as an image.

All of this runs under `torch.inference_mode()`, so no gradients are tracked.

---

## Model Details

| Part | Parameters | Weights file | Trainable |
|---|---|---|---|
| VGG-19 encoder (up to `relu4_1`) | 3.5 M | `vgg_normalised.pth` (80 MB; includes deeper VGG layers that are dropped after loading) | No |
| Decoder | 3.5 M | `decoder_final.pth` (14 MB) | Trained |

- **Speed:** a 512 × 512 content + style pair takes about **5 to 6 seconds** on a 2-thread laptop CPU (PyTorch 2.5, measured locally). A GPU is much faster.
- **CPU threads:** the app caps PyTorch at 4 threads to stay responsive on shared hardware.
- **Training:** the decoder was trained with the training script in [Style_transfer_Vgg](https://github.com/Amanjaiiinnn/Style_transfer_Vgg), which also contains a Flask version of this app. `utils/utils.py` here keeps the dataset loader and transforms used for training (`ImageFolderDataset`, `get_transform`).

---

## Tech Stack

| Area | Tools |
|---|---|
| Model | PyTorch, Torchvision |
| Images | Pillow |
| Interface | Gradio 6 (Blocks, Examples, request queue) |
| Hosting | Hugging Face Spaces, with model weights in Git LFS |

---

## Project Structure

```text
NST/
├── app.py                 # Gradio app: loads models, preprocessing, stylize(), UI
├── README.md              # This file; the header block configures the Space
├── requirements.txt       # torch, torchvision, Pillow, numpy
├── .gitattributes         # Git LFS rules (*.pth and other binary formats)
├── .gitignore
├── utils/
│   ├── models.py          # VGGEncoder and Decoder
│   └── utils.py           # AdaIN, channel mean/std, training dataset + transforms
├── weights/
│   ├── vgg_normalised.pth # Pre-trained VGG-19 weights
│   └── decoder_final.pth  # Trained AdaIN decoder
└── examples/
    ├── content_lenna.jpg
    ├── content_golden_gate.jpg
    ├── style_sketch.png
    └── style_la_muse.jpg
```

---

## Run Locally

### Prerequisites

- Python 3.11 (the version the Space uses)
- [Git LFS](https://git-lfs.com), to download the model weights

### Steps

```bash
git lfs install
git clone https://huggingface.co/spaces/Amanprime/NST
cd NST
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install the dependencies. Gradio isn't listed in `requirements.txt` because Hugging Face installs the version set in the header (`sdk_version`), so install it yourself:

```bash
pip install -r requirements.txt gradio==6.14.0
```

Start the app:

```bash
python app.py
```

Gradio prints a local URL, usually `http://127.0.0.1:7860`.

If either weights file is missing, the app stops at start-up with a `FileNotFoundError` that names the missing file.

---

## Using the App

1. Upload a **content image**, the photo you want to repaint.
2. Upload a **style image**, the artwork whose look you want.
3. Set **Style strength**: `1.0` for the full style, lower values to keep more of the original.
4. Click **Transfer style**.

You can also click one of the examples under the button to try the app instantly.

---

## Deploying Your Own Space

1. Create a new Space on Hugging Face and choose the **Gradio** SDK.
2. Push these files to it. The `.gitattributes` file makes sure the `.pth` weights go through Git LFS.
3. Keep the header block at the top of this README. Hugging Face reads it to pick the SDK version (`6.14.0`), the Python version (`3.11`) and the entry file (`app.py`).

---

## Author

**Aman Jain** · [GitHub](https://github.com/Amanjaiiinnn)
