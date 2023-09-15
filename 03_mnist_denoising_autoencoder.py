import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


class DenoisingAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 1, 28, 28)
        return out


def add_noise(x, noise_level=0.4):
    noise = noise_level * torch.randn_like(x)
    x_noisy = x + noise
    return torch.clamp(x_noisy, 0.0, 1.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = DenoisingAE(latent_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, n_train = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            x_noisy = add_noise(x)

            optimizer.zero_grad()
            x_rec = model(x_noisy)
            loss = criterion(x_rec, x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        train_loss /= n_train

        # quick eval reconstruction loss on test set
        model.eval()
        test_loss, n_test = 0.0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_noisy = add_noise(x)
                x_rec = model(x_noisy)
                loss = criterion(x_rec, x)
                test_loss += loss.item() * x.size(0)
                n_test += x.size(0)
        test_loss /= n_test

        print(f"Epoch {epoch:02d} | train_recon_loss={train_loss:.4f} | test_recon_loss={test_loss:.4f}")

    torch.save(model.state_dict(), "mnist_denoising_ae.pt")
    print("Saved model weights to mnist_denoising_ae.pt")

    # save a small grid of clean / noisy / denoised images to visually show the novelty
    model.eval()
    os.makedirs("figures", exist_ok=True)
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)[:8]
        x_noisy = add_noise(x)
        x_rec = model(x_noisy)

        x = x.cpu()
        x_noisy = x_noisy.cpu()
        x_rec = x_rec.cpu()

        num_imgs = x.size(0)
        fig, axes = plt.subplots(3, num_imgs, figsize=(num_imgs * 2, 6))
        for i in range(num_imgs):
            axes[0, i].imshow(x[i, 0], cmap="gray")
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Clean")

            axes[1, i].imshow(x_noisy[i, 0], cmap="gray")
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Noisy")

            axes[2, i].imshow(x_rec[i, 0], cmap="gray")
            axes[2, i].axis("off")
            if i == 0:
                axes[2, i].set_title("Denoised")

        plt.tight_layout()
        out_path = os.path.join("figures", "mnist_denoising_examples.png")
        plt.savefig(out_path)
        print(f"Saved example reconstructions to {out_path}")


if __name__ == "__main__":
    main()
