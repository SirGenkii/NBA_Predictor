# Dockerfile
FROM jupyter/datascience-notebook:latest

# Mise à jour des paquets Python
RUN pip install --upgrade pip

# Installation PyTorch avec CUDA 11.8 (compatible 3060 Ti)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installation TensorFlow GPU (version compatible CUDA 11.8)
RUN pip install tensorflow==2.15.0

# (Optionnel) Installation d'autres libs utiles
RUN pip install xgboost lightgbm catboost seaborn plotly scikit-optimize

# Vérification CUDA (non bloquante)
RUN python -c "import torch; print('✅ PyTorch GPU dispo:', torch.cuda.is_available())"

WORKDIR /workspace

