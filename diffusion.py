import torch
from config import T

# 前向diffusion计算参数
betas = torch.linspace(0.0001, 0.02, T)  # (T,)
alphas = 1 - betas  # (T,)
alphas_cumprod = torch.cumprod(alphas, dim=-1)  # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)  # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # denoise用的方差   (T,)


# 执行前向加噪
def forward_add_noise(x, t):  # x: (batch,channel,height,width), t: (batch,)
    eps = torch.randn_like(x)  # 为每张图片生成第t步的高斯噪音   (batch,channel,height,width)
    alpha_bar_t = alphas_cumprod[t].view(x.size(0), 1, 1, 1)
    x_t = (
        torch.sqrt(alpha_bar_t) * x
        + torch.sqrt(1 - alpha_bar_t) * eps
    )  # 基于公式直接生成第t步加噪后图片
    return x_t, eps

