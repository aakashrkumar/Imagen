wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/aakash/miniconda3
source ~/miniconda3/bin/activate
pip install -U "jax[tpu]>=0.2.16" jaxlib -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip install -U "jax[tpu]>=0.2.16" jaxlib -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r scripts/requirements.txt
