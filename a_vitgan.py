import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange



# Code based on https://github.com/lucidrains/vit-pytorch
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 1024, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.proj = project_out
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim,2)
        self.layer_norm = nn.LayerNorm(dim,eps=1e-6)
        self.linear1 = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim,self.dim))
        
    def forward(self, inputs):
        attention_out = self.attention(inputs)
        first_out = self.layer_norm(torch.add(attention_out,inputs))
        lin1_out = self.linear1(first_out.cuda())
        return self.layer_norm(torch.add(first_out,lin1_out))

class A_VITGAN(nn.Module):

    def __init__(self, image_size=224, patch_size=16,channels=3,dim=256, dense_units=32):
        super(A_VITGAN, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.dim = dim

        self.dense_units = dense_units
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding_atk_def = nn.Parameter(torch.randn(1, num_patches*2 , dim),requires_grad=True)
        self.pos_embedding_atk = nn.Parameter(torch.randn(1, num_patches , dim),requires_grad=True)
        self.pos_embedding_def = nn.Parameter(torch.randn(1, num_patches , dim),requires_grad=True)
        
        self.final_transpose = nn.Sequential(
            nn.LazyConvTranspose2d(3, 5,dilation=2,output_padding=1,bias=False),
            nn.Upsample(224),
            nn.Tanh(),
        )
        self.convolution= nn.Sequential(
            nn.LazyConvTranspose2d(1024, 5, stride=2),
            nn.LazyBatchNorm2d(),
            nn.Tanh(),
            
            nn.LazyConv2d(1024,3),
            nn.LazyBatchNorm2d(),
            nn.Tanh(),
            nn.LazyConv2d(1024,3),
            nn.LazyBatchNorm2d(),
            nn.Tanh(),
            
            nn.LazyConvTranspose2d(512, 5, stride=2),
            nn.LazyBatchNorm2d(),
            nn.Tanh(),
            
            
            nn.LazyConv2d(256,3),
            nn.LazyBatchNorm2d(),
            nn.Tanh(),
            nn.LazyConv2d(256,3),
            nn.LazyBatchNorm2d(),
            nn.Tanh(),
            
            nn.LazyConvTranspose2d(64, 5, stride=2),
            nn.LazyBatchNorm2d(),
            nn.Tanh(),
            
            
            nn.LazyConv2d(64,3),
            nn.Tanh(),
            nn.LazyConv2d(64,3),
            nn.Tanh(),
            
        )
        self.transformer_block_atk_def_1 = TransformerBlock(dim)
        self.transformer_block_atk_def_2 = TransformerBlock(dim)
        self.transformer_block_atk_def_3 = TransformerBlock(dim)
        self.transformer_block_atk_def_4 = TransformerBlock(dim)
        
        self.transformer_block_atk_1 = TransformerBlock(dim)
        self.transformer_block_atk_2 = TransformerBlock(dim)
        self.transformer_block_atk_3 = TransformerBlock(dim)
        self.transformer_block_atk_4 = TransformerBlock(dim)
        
        self.transformer_block_def_1 = TransformerBlock(dim)
        self.transformer_block_def_2 = TransformerBlock(dim)
        self.transformer_block_def_3 = TransformerBlock(dim)
        self.transformer_block_def_4 = TransformerBlock(dim)

    def generate_attack_images(self, original_images,attack_matrix, epsilon=20):
        for i in range(0,len(original_images)):
            original_images[i] = torch.mul(attack_matrix[i], min(1,epsilon/torch.linalg.norm(attack_matrix[i]))) + original_images[i]
        return original_images    
    
    def forward(self, img_attack, img):
        patches_defense = self.to_patch(img)
        patches_attack = self.to_patch(img_attack)
        patches_attack_defense = torch.cat((patches_defense, patches_attack),dim=1)
        
        #transformer blocks attack+defense tokens
        b, n, _ = patches_attack_defense.shape
        patches_attack_defense += self.pos_embedding_atk_def[:, : n*2]
        x_atk_def = self.transformer_block_atk_def_1(patches_attack_defense)
        x_atk_def = self.transformer_block_atk_def_2(x_atk_def)
        x_atk_def = self.transformer_block_atk_def_3(x_atk_def)
        x_atk_def = self.transformer_block_atk_def_4(x_atk_def)
        x_atk_def = x_atk_def.view(x_atk_def.shape[0],16,16,392)
        
        #transformer blocks attack
        b, n, _ = patches_attack.shape
        patches_attack += self.pos_embedding_atk[:, : n]
        x_atk = self.transformer_block_atk_1(patches_attack)
        x_atk = self.transformer_block_atk_2(x_atk)
        x_atk = self.transformer_block_atk_3(x_atk)
        x_atk = self.transformer_block_atk_4(x_atk)
        x_atk = x_atk.view(x_atk.shape[0],16,16,196)
        
        #transformer blocks defense
        b, n, _ = patches_defense.shape
        patches_defense += self.pos_embedding_def[:, : n]
        x_def = self.transformer_block_def_1(patches_defense)
        x_def = self.transformer_block_def_2(x_def)
        x_def = self.transformer_block_def_3(x_def)
        x_def = self.transformer_block_def_4(x_def)
        x_def = x_def.view(x_def.shape[0],16,16,196)
        
        x = torch.cat((x_atk_def,x_atk,x_def),dim=-1)
        x=self.convolution(x)
        x=self.final_transpose(x)
        x = self.generate_attack_images(img, x)
        return x
        
