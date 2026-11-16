# Chrysalis on Linux with an NVIDIA RTX 30 series or newer

As run on Ubuntu 22.04.

1. Driver: `sudo apt install nvidia-driver-580`, then restart. `nvidia-smi` lists the card when it is in.
2. CUDA toolkit and build tools:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install cuda-toolkit-13-0 build-essential cmake git curl unzip
```

3. Get Chrysalis and start it. The first start builds the engines for the card, a few minutes:

```sh
git clone https://github.com/michaelDinelle/chrysalis-inference
cd chrysalis-inference
./chrysalis.sh
```
