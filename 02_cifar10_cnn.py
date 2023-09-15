import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """Basic CNN for CIFAR-10 with ~120k parameters."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_loaders(batch_size=128, use_augmentation=True):
    if use_augmentation:
        train_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


def train_eval(model, train_loader, test_loader, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss, train_correct, n_train = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            n_train += y.size(0)

        train_loss /= n_train
        train_acc = train_correct / n_train

        # eval
        model.eval()
        test_loss, test_correct, n_test = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                test_loss += loss.item() * x.size(0)
                test_correct += (logits.argmax(1) == y).sum().item()
                n_test += y.size(0)

        test_loss /= n_test
        test_acc = test_correct / n_test

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Toggle this flag to show the effect of data augmentation.
    use_augmentation = True

    train_loader, test_loader = get_loaders(use_augmentation=use_augmentation)
    model = SimpleCNN().to(device)

    train_eval(model, train_loader, test_loader, device)

    suffix = "aug" if use_augmentation else "noaug"
    torch.save(model.state_dict(), f"cifar10_cnn_{suffix}.pt")
    print(f"Saved weights to cifar10_cnn_{suffix}.pt")


if __name__ == "__main__":
    main()
