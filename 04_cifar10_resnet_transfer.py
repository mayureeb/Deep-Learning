import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_loaders(batch_size=128):
    train_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


def build_resnet(num_classes=10, finetune_last_layer_only=True):
    # uses ImageNet-pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if finetune_last_layer_only:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def train_eval(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
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

    train_loader, test_loader = get_loaders()

    # First run: linear probe (only last layer trained)
    print("=== Linear probe (only final layer trained) ===")
    model_linear = build_resnet(num_classes=10, finetune_last_layer_only=True).to(device)
    train_eval(model_linear, train_loader, test_loader, device, epochs=3, lr=1e-3)
    torch.save(model_linear.state_dict(), "cifar10_resnet18_linear_probe.pt")
    print("Saved linear probe weights to cifar10_resnet18_linear_probe.pt")

    # Second run: fine-tune full network (unfreeze and continue)
    print("=== Full fine-tuning (all layers trainable) ===")
    for param in model_linear.parameters():
        param.requires_grad = True
    train_eval(model_linear, train_loader, test_loader, device, epochs=2, lr=1e-4)
    torch.save(model_linear.state_dict(), "cifar10_resnet18_finetuned.pt")
    print("Saved fine-tuned weights to cifar10_resnet18_finetuned.pt")


if __name__ == "__main__":
    main()
