# Autors: Sergejs Kiseļovs
# CycleGAN modeļa apmācība

import os
import torch
import itertools
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Izmanto ierīci: {device}")

print("GPU: ", torch.cuda.is_available())


main_dataset = "3d-animation 5000"
films_dataset = "Films 5000"

# Parametri
img_width = 640
img_height = 360
batch_size = 1
num_epochs = 100
learning_rate = 0.0002
beta1 = 0.5
save_path = f'/root/CartoonCycleGAN/{main_dataset} saved_images/'
model_path = f'/root/CartoonCycleGAN/{main_dataset} saved_models/'

# Izveido nepieciešamās mapes, ja tās neeksistē
print("Izveido nepieciešamās mapes, ja tās neeksistē")
os.makedirs(save_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Datu ielāde ar jauniem attēlu izmēriem
print("Datu ielāde ar jauniem attēlu izmēriem")
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # Atjauninātie izmēri
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(f"{root_dir}/*")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Otrais elements - fiktīvs klases numurs

print(f"Cartoon datu kopums: /root/CartoonCycleGAN/{main_dataset}")
if not os.path.exists(f'/root/CartoonCycleGAN/{main_dataset}'):
    print(f"Mapa nav atrasta: /root/CartoonCycleGAN/{main_dataset}")

# Izmanto šo klasi datu ielādei
print("Izmanto šo klasi datu ielādei")
train_cartoon = CustomImageDataset(f'/root/CartoonCycleGAN/{main_dataset}', transform=transform)
train_photo = CustomImageDataset(f'/root/CartoonCycleGAN/{films_dataset}', transform=transform)
test_cartoon = CustomImageDataset(f'/root/CartoonCycleGAN/Test dataset/test_cartoon', transform=transform)
test_photo = CustomImageDataset(f'/root/CartoonCycleGAN/Test dataset/test_photo', transform=transform)

print(f"Vai transformācijas ir piemērotas? Attēlu skaits: {len(train_cartoon)}")

train_loader_cartoon = DataLoader(train_cartoon, batch_size=batch_size, shuffle=True)
train_loader_photo = DataLoader(train_photo, batch_size=batch_size, shuffle=True)
test_loader_cartoon = DataLoader(test_cartoon, batch_size=1, shuffle=False)
test_loader_photo = DataLoader(test_photo, batch_size=1, shuffle=False)

# Ģeneratora definēšana
print("Ģeneratora definēšana")
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Diskriminatora definēšana
print("Diskriminatora definēšana")
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)



# Ceļš uz failiem ar svaru datiem
path_G_photo2cartoon = f"{model_path}/G_photo2cartoon.pth"
path_G_cartoon2photo = f"{model_path}/G_cartoon2photo.pth"
path_D_photo = f"{model_path}/D_photo.pth"
path_D_cartoon = f"{model_path}/D_cartoon.pth"

# Tīklu inicializācija
print("Tīklu inicializācija")
G_photo2cartoon = Generator().to(device)
G_cartoon2photo = Generator().to(device)
D_photo = Discriminator().to(device)
D_cartoon = Discriminator().to(device)

# Ielādē saglabātos svaru datus, ja tie eksistē
if os.path.exists(path_G_photo2cartoon):
    G_photo2cartoon.load_state_dict(torch.load(path_G_photo2cartoon))
    print("Ielādēti svaru dati G_photo2cartoon")

if os.path.exists(path_G_cartoon2photo):
    G_cartoon2photo.load_state_dict(torch.load(path_G_cartoon2photo))
    print("Ielādēti svaru dati G_cartoon2photo")

if os.path.exists(path_D_photo):
    D_photo.load_state_dict(torch.load(path_D_photo))
    print("Ielādēti svaru dati D_photo")

if os.path.exists(path_D_cartoon):
    D_cartoon.load_state_dict(torch.load(path_D_cartoon))
    print("Ielādēti svaru dati D_cartoon")

# Optimizatoru un zaudējuma funkciju definēšana
print("Optimizatoru un zaudējuma funkciju definēšana")
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizatoru inicializācija
optimizer_G = optim.Adam(itertools.chain(G_photo2cartoon.parameters(), G_cartoon2photo.parameters()), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D_photo = optim.Adam(D_photo.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D_cartoon = optim.Adam(D_cartoon.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Ielādē optimizatoru stāvokļus, ja tie eksistē
path_optimizer_G = f"{model_path}/optimizer_G.pth"
path_optimizer_D_photo = f"{model_path}/optimizer_D_photo.pth"
path_optimizer_D_cartoon = f"{model_path}/optimizer_D_cartoon.pth"

if os.path.exists(path_optimizer_G):
    optimizer_G.load_state_dict(torch.load(path_optimizer_G))
    print("Ielādēts optimizatora stāvoklis G")

if os.path.exists(path_optimizer_D_photo):
    optimizer_D_photo.load_state_dict(torch.load(path_optimizer_D_photo))
    print("Ielādēts optimizatora stāvoklis D_photo")

if os.path.exists(path_optimizer_D_cartoon):
    optimizer_D_cartoon.load_state_dict(torch.load(path_optimizer_D_cartoon))
    print("Ielādēts optimizatora stāvoklis D_cartoon")




# Modeļa apmācība
print("Modeļa apmācība")
for epoch in range(0, num_epochs):
#for epoch in range(num_epochs):

    for i, ((photo, _), (cartoon, _)) in enumerate(zip(train_loader_photo, train_loader_cartoon)):
        # Ģeneratoru un diskriminatoru apmācība abos virzienos (foto -> multfilma un multfilma -> foto)

        photo = photo.to(device)
        cartoon = cartoon.to(device)

        # Ģenerators Foto -> Multfilma
        optimizer_G.zero_grad()
        fake_cartoon = G_photo2cartoon(photo)
        #loss_GAN_photo2cartoon = criterion_GAN(D_cartoon(fake_cartoon), torch.ones_like(D_cartoon(fake_cartoon)))
        loss_GAN_photo2cartoon = criterion_GAN(D_cartoon(fake_cartoon), torch.ones_like(D_cartoon(fake_cartoon)).to(device))

        loss_cycle_photo = criterion_cycle(G_cartoon2photo(fake_cartoon), photo) * 10.0
        loss_G = loss_GAN_photo2cartoon + loss_cycle_photo
        loss_G.backward()
        optimizer_G.step()

        # Diskriminators Multfilma
        optimizer_D_cartoon.zero_grad()
        #loss_D_cartoon_real = criterion_GAN(D_cartoon(cartoon), torch.ones_like(D_cartoon(cartoon)))
        loss_D_cartoon_real = criterion_GAN(D_cartoon(cartoon), torch.ones_like(D_cartoon(cartoon)).to(device))

        loss_D_cartoon_fake = criterion_GAN(D_cartoon(fake_cartoon.detach()), torch.zeros_like(D_cartoon(fake_cartoon)))
        loss_D_cartoon = (loss_D_cartoon_real + loss_D_cartoon_fake) * 0.5
        loss_D_cartoon.backward()
        optimizer_D_cartoon.step()

        # Ģenerators Multfilma -> Foto
        optimizer_G.zero_grad()
        fake_photo = G_cartoon2photo(cartoon)
        loss_GAN_cartoon2photo = criterion_GAN(D_photo(fake_photo), torch.ones_like(D_photo(fake_photo)))
        loss_cycle_cartoon = criterion_cycle(G_photo2cartoon(fake_photo), cartoon) * 10.0
        loss_G = loss_GAN_cartoon2photo + loss_cycle_cartoon
        loss_G.backward()
        optimizer_G.step()

        # Diskriminators Foto
        optimizer_D_photo.zero_grad()
        loss_D_photo_real = criterion_GAN(D_photo(photo), torch.ones_like(D_photo(photo)))
        loss_D_photo_fake = criterion_GAN(D_photo(fake_photo.detach()), torch.zeros_like(D_photo(fake_photo)))
        loss_D_photo = (loss_D_photo_real + loss_D_photo_fake) * 0.5
        loss_D_photo.backward()
        optimizer_D_photo.step()

    print()
    print(f"Epohas [{epoch+1}/{num_epochs}] pabeigta")

    if (epoch+1) % 5 == 0:

        # Saglabā testa attēlu rezultātus
        print("Saglabā testa attēlu rezultātus")
        with torch.no_grad():
            for i, (test_photo, _) in enumerate(test_loader_photo):
                test_photo = test_photo.to(device)
                fake_cartoon = G_photo2cartoon(test_photo)
                save_image(fake_cartoon, f"{save_path}/test_photo_to_cartoon_{i}_{epoch + 1}.png")

            for i, (test_cartoon, _) in enumerate(test_loader_cartoon):
                test_cartoon = test_cartoon.to(device)
                fake_photo = G_cartoon2photo(test_cartoon)
                save_image(fake_photo, f"{save_path}/test_cartoon_to_photo_{i}_{epoch + 1}.png")

        # Modeļa saglabāšana
        print("Modeļa saglabāšana")
        torch.save(G_photo2cartoon.state_dict(), f"{model_path}/{epoch + 1}_G_photo2cartoon.pth")
        torch.save(G_cartoon2photo.state_dict(), f"{model_path}/{epoch + 1}_G_cartoon2photo.pth")
        torch.save(D_photo.state_dict(), f"{model_path}/{epoch + 1}_D_photo.pth")
        torch.save(D_cartoon.state_dict(), f"{model_path}/{epoch + 1}_D_cartoon.pth")

        # Optimizatoru stāvokļu saglabāšana
        torch.save(optimizer_G.state_dict(), f"{model_path}/{epoch + 1}_optimizer_G.pth")
        torch.save(optimizer_D_photo.state_dict(), f"{model_path}/{epoch + 1}_optimizer_D_photo.pth")
        torch.save(optimizer_D_cartoon.state_dict(), f"{model_path}/{epoch + 1}_optimizer_D_cartoon.pth")

        torch.save(G_photo2cartoon.state_dict(), f"{model_path}/G_photo2cartoon.pth")
        torch.save(G_cartoon2photo.state_dict(), f"{model_path}/G_cartoon2photo.pth")
        torch.save(D_photo.state_dict(), f"{model_path}/D_photo.pth")
        torch.save(D_cartoon.state_dict(), f"{model_path}/D_cartoon.pth")

        # Optimizatoru stāvokļu saglabāšana
        torch.save(optimizer_G.state_dict(), f"{model_path}/optimizer_G.pth")
        torch.save(optimizer_D_photo.state_dict(), f"{model_path}/optimizer_D_photo.pth")
        torch.save(optimizer_D_cartoon.state_dict(), f"{model_path}/optimizer_D_cartoon.pth")

print("Apmācība pabeigta, rezultāti saglabāti.")
