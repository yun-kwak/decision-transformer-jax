import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
