import numpy as np
import random
from IPython.display import Audio, display
from transformers import AutoFeatureExtractor
from huggingface_hub import notebook_login


notebook_login()

model_checkpoint = "facebook/wav2vec2-base"
batch_size = 32

