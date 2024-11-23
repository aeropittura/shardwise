#!/usr/bin/env python3

"""
Profiler with snakeviz for probing inference / training call stack

Run via Jupyter
"""

from benchmark import ModelBenchmarks


# get_ipython().run_line_magic("load_ext", "snakeviz")


model_benchmarks = ModelBenchmarks()
# for logistic regression select 0
model = model_benchmarks.MODELS[1]
print(model.name)
model_crypten = model.crypten(model_benchmarks.n_features).encrypt()

# profile training
# get_ipython().run_cell_magic(
#     "snakeviz", "", "\nmodel_benchmarks.train_crypten(model_crypten)"
# )

# profile inference
x_enc = model_benchmarks.x_enc
model_crypten.train = False

# get_ipython().run_cell_magic("snakeviz", "", "\n\nmodel_crypten(x_enc)")
