# CS 182 Final Project

[CycleGAN](https://junyanz.github.io/CycleGAN/) is a deep learning architecture for image-to-image style translation. In this project, we reimplement it using the [JAX](https://jax.readthedocs.io/en/latest/) framework.

## Running the Model

```sh
python main.py --train
```

or

```sh
python main.py --predict ./example.jpg
```

## Installing Dependencies

TODO: script to automatically install dependencies?

```sh
conda create -n cs182-proj python=3.9
conda activate cs182-proj
```

If CUDA is available:

```sh
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Otherwise:

```sh
pip install --upgrade "jax[cpu]"
```

Finally:

```sh
pip install flax
pip install torch torchvision
pip install jupyter
```

## Downloading Data

Grant execution access to script, download data and unzip

```sh
chmod +x ./get-horse2zebra.sh
./get-horse2zebra.sh
```
