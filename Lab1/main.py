import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models import get_model  
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers for data loader")
    parser.add_argument("--log", action="store_true", help="Enables logging of the loss and accuracy metrics to Weights & Biases")

    return parser.parse_args()

def get_dataloaders(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
    
def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{c.epochs} [Train]", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            'loss': f"{running_loss / (progress_bar.n + 1):.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, desc="Validation"):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=desc, leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'loss': f"{val_loss / (progress_bar.n + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    args = parse_args()
    
    if args.log:
        import wandb
        wandb.init(project="cifar10-classification", name=f"{args.model}", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")


    print("=> Creating model")
    model = get_model(args.model).to(device)   
    if args.log:
        wandb.watch(model, log="all")
        
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("=> Get dataloaders")
    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)
    
    best_val_acc = 0.0
    
    print("=> Starting training")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if args.log:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        print(f"Leaning Rate: {scheduler.get_last_lr()[0]}")
         
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.model}_best.pth")
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
            
        scheduler.step()
    
    print("\n=> Final test")
    model.load_state_dict(torch.load(f"{args.model}_best.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, device, desc="Test")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
    
    if args.log:
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        # wandb.save(f"{args.model}_best.pth")
        wandb.finish()

if __name__ == '__main__':
    main()
    
# python main.py --model cnn --epochs 50 --batch-size 128 --lr 0.1 --num-workers 2 --log
# python main.py --model mlp --epochs 50 --batch-size 128 --lr 0.1 --num-workers 2 --log
# python main.py --model resnet18 --epochs 50 --batch-size 128 --lr 0.1 --num-workers 2 --log


 