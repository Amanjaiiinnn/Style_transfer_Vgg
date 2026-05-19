## Demo

Hugging Face Space:  
`https://huggingface.co/spaces/Amanprime/NST`

This project applies Neural Style Transfer using the AdaIN method.
It takes two images:

1- Content Image

2- Style Image

and generates a new image where the content image is redrawn in the style of the style image.


## Features

- Upload any content image
- Upload any style/reference image
- Adjust style strength using a slider
- Generate stylized output using PyTorch
- Simple Gradio web interface
- Deployable on Hugging Face Spaces

## Tech Stack

- Python
- PyTorch
- Torchvision
- Pillow
- Gradio
- AdaIN Neural Style Transfer

## Project Structure

NST/
  app.py
  requirements.txt
  README.md
  .gitattributes

  utils/
    models.py
    utils.py

  weights/
    vgg_normalised.pth
    decoder_final.pth



App needs these two model files:
 
weights/vgg_normalised.pth



weights/decoder_final.pth
