import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    """Simple fully-connected network for MNIST."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_loaders(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    n = 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += y.size(0)

    return total_loss / n, correct / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders()
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print("Saved model weights to mnist_mlp.pt")


if __name__ == "__main__":
    main()
