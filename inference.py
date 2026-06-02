import torch
from config import T
from dit import DiT
import matplotlib.pyplot as plt
from diffusion import alphas, alphas_cumprod, variance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备


def backward_denoise(model, x, y):
    steps = [
        x.clone(),
    ]

    x = x.to(DEVICE)
    alpha = alphas.to(DEVICE)
    alpha_bar = alphas_cumprod.to(DEVICE)
    sigma2 = variance.to(DEVICE)
    y = y.to(DEVICE)

    model.eval()
    batch_size = x.size(0)
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((batch_size,), time).to(DEVICE)

            # 预测x_t时刻的噪音
            eps_theta = model(x, t, y)

            # 生成t-1时刻的图像
            shape = (batch_size, 1, 1, 1)
            alpha_t = alpha[t].view(*shape)
            alpha_bar_t = alpha_bar[t].view(*shape)
            mu_theta = 1 / torch.sqrt(alpha_t) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta
            )
            if time != 0:
                sigma_t = torch.sqrt(sigma2[t].view(*shape))
                x = mu_theta + torch.randn_like(x) * sigma_t
            else:
                x = mu_theta
            x = torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x)
    return steps


def main():
    model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(DEVICE)  # 模型
    model.load_state_dict(torch.load("model.pth"))

    # 生成噪音图
    batch_size = 10
    x = torch.randn(size=(batch_size, 1, 28, 28))  # (5,1,24,24)
    y = torch.arange(start=0, end=10, dtype=torch.long)  #
    # 逐步去噪得到原图
    steps = backward_denoise(model, x, y)
    # 绘制数量
    num_imgs = 20
    # 绘制还原过程
    plt.figure(figsize=(15, 15))
    for b in range(batch_size):
        for i in range(0, num_imgs):
            idx = int(T / num_imgs) * (i + 1)
            # 像素值还原到[0,1]
            final_img = (steps[idx][b].to("cpu") + 1) / 2
            # tensor转回PIL图
            final_img = final_img.permute(1, 2, 0)
            plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
            plt.imshow(final_img)
    plt.show()


if __name__ == "__main__":
    main()
