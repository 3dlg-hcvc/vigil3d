# apt-get install libsparsehash-dev ffmpeg libsm6 libxext6 libopenblas-dev

pip install -e .
pip install --ignore-installed -U -r requirements.txt

pip install torch==2.2.1 torchvision==0.17.1 torch_geometric==2.5.3 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -U git+https://github.com/atwang16/MinkowskiEngine.git
MAX_JOBS=2 pip install -U git+https://github.com/mit-han-lab/torchsparse.git
MAX_JOBS=2 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install "pointnet2_ops@git+https://github.com/atwang16/pointnet2.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

python -m spacy download en_core_web_md