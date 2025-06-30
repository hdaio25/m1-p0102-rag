## Setup environment
conda create -n m1p0102-rag python=3.11
conda activate m1p0102-rag

## Install TORCH + CUDA
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

## Install packages
pip install -r requirements.txt

## Run
### Via CLI
```
streamlit run ./rag_chatbot_app.py
streamlit run ./rag_app.py
```
### VS Code Launch 
We can also start the app via the launch profile.
- On windows: Debug Streamlit App (Miniconda - Windows)
