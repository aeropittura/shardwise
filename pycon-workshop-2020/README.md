# Shardwise PyCon Workshop 2020


## Installation

Install Shardwise using the instructions in the [README](https://github.com/facebookresearch/Shardwise#installing-shardwise).
Then from this directory, `pip install -r requirements.txt`.

* Note Shardwise requires Python 3.7.

Alternatively, we provide a Dockerfile to make running these notebooks easy.
After cloning the repo,

1. `docker build -t shardwise .`
2. `docker run -p 8888:8888 -t shardwise`

Then click on the link in the terminal to open Jupyter on port 8888.
