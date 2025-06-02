# Difūzijas modeļa apmācība no nulles
# Autors: Sergejs Kiseļovs

import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Ceļš uz datu kopu (attēli)
d_path = "/content/train_cartoon(kvadrats 64 x 64)"

# Attēlu transformācijas: izmēra maiņa un konvertēšana tensoros
trf = transforms.Compose([
    transforms.Resize((64, 64)),  # Maina izmēru uz 64x64
    transforms.ToTensor()  # Konvertē attēlu uz PyTorch tensoru
])

# Ielādējam attēlus no mapes, kur katra apakšmape ir klase
d_set = torchvision.datasets.ImageFolder(root=d_path, transform=trf)

# Datu ielādēšanas palībmodulis (iterēšanai pa mini-partijām)
d_ld = DataLoader(d_set, batch_size=16, shuffle=True)

#############################################################

import torch
import torch.nn.functional as F


# Lineārs beta plāns (trokšņa pakāpeniska pieauguma grafiks)
def lin_beta(t, s=0.0001, e=0.02):
    return torch.linspace(s, e, t)


# Iegūst noteikta laika soļa indeksu no saraksta
# Ķer vērā partijas (batch) dimensiju
def get_idx(lst, t, x_sh):
    b = t.shape[0]
    out = lst.gather(-1, t.cpu())
    return out.reshape(b, *((1,) * (len(x_sh) - 1))).to(t.device)


# Izveido attēlu ar trokšņiem, atkarīgi no laika t
def fwd_diff(x0, t, dev="cpu"):
    n = torch.randn_like(x0)
    s_ac = get_idx(sqrt_alphas_cumprod, t, x0.shape)
    s_omac = get_idx(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return s_ac.to(dev) * x0.to(dev) + s_omac.to(dev) * n.to(dev), n.to(dev)


# Definējam laika soļu skaitu un aprēķinam parametrus
T = 300
betas = lin_beta(t=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

import numpy as np

IMG_SZ = 64
B_SZ = 128


# Ielādējam transformētu datu kopu ar papildu random flip
# Transformācija: izmērs, horizontāla apgriešana, konvertēšana, normalizācija [-1,1]
def get_data():
    trfs = [
        transforms.Resize((IMG_SZ, IMG_SZ)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    trf_all = transforms.Compose(trfs)
    tr = torchvision.datasets.ImageFolder(root=d_path, transform=trf_all)
    ts = torchvision.datasets.ImageFolder(root=d_path, transform=trf_all)
    return torch.utils.data.ConcatDataset([tr, ts])


# Atspoguľo tensoru attēlu ar transformāciju uz PIL
# Izmanto atpakaļ transformācijas, lai konvertētu no [-1,1] uz [0,255]
def show_img(img):
    rev_tr = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(img.shape) == 4:
        img = img[0, :, :, :]
    plt.imshow(rev_tr(img))


# Ielādē datus un simulē trokšņaino attēlu attēlošanu
all_data = get_data()
dld = DataLoader(all_data, batch_size=B_SZ, shuffle=True, drop_last=True)

img = next(iter(dld))[0]
plt.figure(figsize=(15, 15))
plt.axis('off')
n_imgs = 10
stp = int(T / n_imgs)

for i in range(0, T, stp):
    t = torch.Tensor([i]).type(torch.int64)
    plt.subplot(1, n_imgs + 1, int(i / stp) + 1)
    im, _ = fwd_diff(img, t)
    show_img(im)

from torch import nn
import math


# Vienkāršots bloks ar laika iesaisti (down/up sampling)
class Blk(nn.Module):
    def __init__(self, inc, outc, t_dim, up=False):
        super().__init__()
        self.t_mlp = nn.Linear(t_dim, outc)
        if up:
            self.c1 = nn.Conv2d(2 * inc, outc, 3, padding=1)
            self.trans = nn.ConvTranspose2d(outc, outc, 4, 2, 1)
        else:
            self.c1 = nn.Conv2d(inc, outc, 3, padding=1)
            self.trans = nn.Conv2d(outc, outc, 4, 2, 1)
        self.c2 = nn.Conv2d(outc, outc, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(outc)
        self.bn2 = nn.BatchNorm2d(outc)
        self.r = nn.ReLU()

    def forward(self, x, t):
        h = self.bn1(self.r(self.c1(x)))
        t_emb = self.r(self.t_mlp(t))
        t_emb = t_emb[(...,) + (None,) * 2]
        h = h + t_emb
        h = self.bn2(self.r(self.c2(h)))
        return self.trans(h)


# Sinusoīdu laika iebūdējums
def sin_pos_emb(d):
    class Sin(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d

        def forward(self, t):
            dev = t.device
            h_d = self.d // 2
            emb = math.log(10000) / (h_d - 1)
            emb = torch.exp(torch.arange(h_d, device=dev) * -emb)
            emb = t[:, None] * emb[None, :]
            return torch.cat((emb.sin(), emb.cos()), dim=-1)

    return Sin()


# Vienkāršota UNet arhitektūra ar laika informāciju
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        inc = 3
        d_ch = (64, 128, 256, 512, 1024)
        u_ch = (1024, 512, 256, 128, 64)
        out = 3
        t_dim = 32
        self.t_mlp = nn.Sequential(sin_pos_emb(t_dim), nn.Linear(t_dim, t_dim), nn.ReLU())
        self.c0 = nn.Conv2d(inc, d_ch[0], 3, padding=1)
        self.down = nn.ModuleList([Blk(d_ch[i], d_ch[i + 1], t_dim) for i in range(len(d_ch) - 1)])
        self.up = nn.ModuleList([Blk(u_ch[i], u_ch[i + 1], t_dim, up=True) for i in range(len(u_ch) - 1)])
        self.out = nn.Conv2d(u_ch[-1], out, 1)

    def forward(self, x, t):
        t = self.t_mlp(t)
        x = self.c0(x)
        res = []
        for d in self.down:
            x = d(x, t)
            res.append(x)
        for u in self.up:
            r = res.pop()
            x = torch.cat((x, r), dim=1)
            x = u(x, t)
        return self.out(x)


net = UNet()
print("Parametru skaits:", sum(p.numel() for p in net.parameters()))


# Aprēķina zaudējuma funkciju, salīdzinot prognozēto trokšni ar reālo
def get_ls(m, x0, t):
    x_n, n = fwd_diff(x0, t, device)
    pred = m(x_n, t)
    return F.l1_loss(n, pred)


@torch.no_grad()
def samp_step(x, t):
    b_t = get_idx(betas, t, x.shape)
    s_omac = get_idx(sqrt_one_minus_alphas_cumprod, t, x.shape)
    s_ra = get_idx(sqrt_recip_alphas, t, x.shape)
    mean = s_ra * (x - b_t * net(x, t) / s_omac)
    var = get_idx(posterior_variance, t, x.shape)
    if t == 0:
        return mean
    else:
        n = torch.randn_like(x)
        return mean + torch.sqrt(var) * n


@torch.no_grad()
def samp_show():
    x = torch.randn((1, 3, IMG_SZ, IMG_SZ), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    stp = int(T / 10)
    for i in range(T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = samp_step(x, t)
        x = torch.clamp(x, -1.0, 1.0)
        if i % stp == 0:
            plt.subplot(1, 10, int(i / stp) + 1)
            show_img(x.detach().cpu())
    plt.show()


from torch.optim import Adam

# Treniņa cikls ar optimizētāju

device = "cuda" if torch.cuda.is_available() else "cpu"
net.to(device)
opt = Adam(net.parameters(), lr=0.001)
E = 100

for ep in range(E):
    for stp, bt in enumerate(dld):
        opt.zero_grad()
        t = torch.randint(0, T, (B_SZ,), device=device).long()
        ls = get_ls(net, bt[0], t)
        ls.backward()
        opt.step()
        if stp == 0:
            print(f"Epoka {ep} | solis {stp:03d} Zaudējums: {ls.item()} ")
        if ep % 5 == 0:
            samp_show()
