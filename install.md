conda create --name deepFace python=3.6

conda activate deepFakes

conda install -y -c pytorch pytorch torchvision cudatoolkit=10.1
conda install -y -c conda-forge matplotlib
conda install -y pandas

pip install opencv-python

mkdir checkpoints
