# Dataset placement

This directory intentionally contains no patient or benchmark images.

Download the official MedMNIST v2 archives from <https://medmnist.com/> and
place the following unmodified files here:

- `bloodmnist.npz`
- `organamnist.npz`
- `pathmnist.npz`

The experiment configuration points to these exact relative paths. The loader
validates array shapes, channels, labels, and native resolution before creating
a local memory-mapped cache under `.cache/medmnist/`.
