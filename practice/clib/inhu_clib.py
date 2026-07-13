from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = "./hf_models"

class InhuClib(nn.Module):
    def __init__(self, i_encoder:nn.Module, t_encoder:nn.Module, 
                 tokenizer:PreTrainedTokenizerBase, output_dim:int = 256):
        super().__init__()
        
        self.i_encoder = i_encoder
        self.t_encoder = t_encoder
        vision_hidden_dim:int = i_encoder.config.hidden_size
        text_hidden_dim:int = t_encoder.config.hidden_size
        
        self.i_projection = nn.Linear(vision_hidden_dim, output_dim, bias=False)
        self.t_projection = nn.Linear(text_hidden_dim, output_dim, bias=False)
        
        self.logit_scale = nn.Parameter(torch.tensor(torch.exp).log())
    
    def encode_i(self, pixel_values:Tensor) -> Tensor:
        image_vector = self.i_encoder(pixel_values)
        image_features = image_vector.pooler_output
        
        embedded_image = self.i_projection(image_features)
        embedded_image = F.normalize(embedded_image, dim = -1)
        
        return embedded_image
    
    def encode_t(self, input_ids:Tensor, attention_mask:Tensor):
        text_vector = self.t_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_vector.pooler_output
        
        embedded_text = self.t_projection(text_features)
        embedded_text = F.normalize(embedded_text, dim=-1)
        
        return embedded_text
        
    def forward(self, img:Tensor, input_ids:Tensor, attention_mask:Tensor):
        image_embeddings = self.encode_i(img)
        text_embeddings = self.encode_t(input_ids, attention_mask)
        logits_per_image = self.logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
    
    def train_InhuClib(self, train_loader:DataLoader, test_loader:DataLoader, num_epoch:int=10) -> Self:
        logit_image, logit_text = model(pixel_values, )
        return self