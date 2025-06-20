uv pip install tqdm
uv pip install numpy
uv pip install pandas
uv pip install matplotlib
uv pip install seaborn
uv pip install scikit-learn
uv pip install jupyter

# pytorch
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# uv pip install torch torchvision torchaudio

# ubuntu
# uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install --upgrade pip && pip install "unsloth[cu118-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
