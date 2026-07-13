import torch
import torch.nn as nn
from torch import Tensor

class InhuBatchEmbedding(nn.Module):
    def __init__(self, image_size:int=32, patch_size:int=4, in_channels:int=3, embed_dim:int=128):
        super().__init__()
        
        assert image_size % patch_size == 0
        
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
    
    def forward(self, X:Tensor):
        batch_size = X.shape[0]
        
        X = self.projection(X) #(batch, embed_dim, patch_vertical, patch_horizontal)
        X = X.flatten(2) #(batch, embed_dim, patch_vertical * patch_horizontal)
        X = X.transpose(1,2) #(batch, patch_vertical * patch_horizontal, embed_dim) transformer expect (batch, seq, feature_dim)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat([cls_token, X], dim=1)
        X = X + self.position_embedding
        
        return X

class InhuMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim:int = 128, num_heads:int = 8, dropout:float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim:int = embed_dim//num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, X:Tensor):
        batch_size, num_token, embed_dim = X.shape
        
        q:Tensor = self.q_proj(X)
        k:Tensor = self.k_proj(X)
        v:Tensor = self.v_proj(X)
        
        q = q.reshape(batch_size, num_token, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, num_token, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, num_token, self.num_heads, self.head_dim)
        
        # shape = [batch_size, self.num_heads, num_token, self.head_dim]
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        attention_score = (q @ k.transpose(-1,-2)) / self.head_dim ** 0.5
        attention_weight = torch.softmax(attention_score, dim = -1)
        attention_weight = self.dropout(attention_weight)
        attention_output = attention_weight @ v
        
        attention_output = attention_output.transpose(1,2)
        attention_output = attention_output.reshape(
            batch_size, num_token, self.embed_dim)
        
        output = self.output(attention_output)
        output = self.dropout(output)
        return output

class InhuVitEncoder(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, mlp_dim:int, dropout:float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = InhuMultiheadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, X):
        X = X + self.attention(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X

class InhuVit(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, in_channels: int = 3, 
                 embed_dim: int = 128, num_heads: int = 8, mlp_dim: int = 512, 
                 num_layers: int = 4, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = InhuBatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.encoder_list = nn.ModuleList([InhuVitEncoder(embed_dim, num_heads, mlp_dim, dropout)
                                           for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, X:Tensor) ->Tensor:
        X = self.embedding(X)
        
        for encoder in self.encoder_list:
            X = encoder(X)
        
        X = self.norm(X)
        
        cls_output = X[:, 0]
        logit = self.classifier(cls_output)
        
        return logit