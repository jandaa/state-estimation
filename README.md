# state-estimation

This project contains two implementations of an assignment I did for my state estimation course. The original is written in Python and the second is a re-implementation in Julia for the purposes of exploring the language. 

For more details, please visit my [project page](http://www.ajanda.ca/projects/julia)

# Setup

This setup assumes you already have python 3.8 or greater and julia already installed.
## Download data

First, we need to download the dataset used for the project. To do this on a unix system, use:

```bash
wget https://files.ajanda.ca/state-estimation/data.mat
```

## Environments
Python can be setup using the convenience script:

```bash
bash set_python_env.bash
```

## Running Julia

```bash
julia src/main.jl
```

## Running Python

```bash
python src/main.py
```