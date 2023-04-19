# How to Run Script

ensure to run on Python 3.9

Galactica is quite large; create the following cache variables to handle model downloads
export HUGGINGFACE_HUB_CACHE=/project/HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE=/project/HUGGINGFACE_HUB_CACHE
export PIP_CACHE_DIR=/project/PIP_CACHE

pip install galai torch==1.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html --no-cache-dir
