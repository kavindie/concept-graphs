## Download data
```[bash]
$ wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
$ unzip Replica.zip 
```

## Create virtual env
```[bash]
$ module load python/3.12.0
$ module load cuda/12.4.0 
$ python3 -m venv .venv
$ . .venv/bin/activate
$ pip install --upgrade pip
$ pip install pip-tools
$ pip install -e .
$ pip install torch
$ pip install torchvision
$ pip install torchaudio
$ pip install "git+https://github.com/facebookresearch/pytorch3d.git" #pytorch3d
$ pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy ultralytics imageio supervision natsort
$ CD go to Grounded SAM and  pip install -e GroundingDINO
$ export GSA_PATH=/scratch3/kat049/Grounded-Segment-Anything
export PYTHONPATH=$PYTHONPATH:../Grounded-Segment-Anything/segment_anything/
```
## Grounded SAM
I had it git cloned
```
$ pip install -q -r requirements.txt
$ cd /content/Grounded-Segment-Anything/GroundingDINO
$ pip install -q .
$ cd /content/Grounded-Segment-Anything/segment_anything
$ pip install -q .
$ cd /content/Grounded-Segment-Anything

```