import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import seaborn as sns
import pandas as pd

from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # 4. Change the classifier head
    class_names = ['angry', 'happy', 'neutral', 'sad', 'surprise']

    set_seeds()
    pretrained_vit.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=512),
        nn.ReLU(),
        nn.Dropout(p=0.75),
        nn.BatchNorm1d(512),  # Add Batch Normalization
        nn.Linear(in_features=512, out_features=len(class_names))
    ).to(device)

    # Setup directory paths to train and test images
    train_dir = 'D:/Tools/AI Training datasets/Custom/train'
    test_dir = 'D:/Tools/AI Training datasets/Custom/test'

    # Get automatic transforms from pretrained ViT weights
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    train_transform = transforms.Compose([
        pretrained_vit_transforms,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=3)
    ])

    import os

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    NUM_WORKERS = 4


    def create_dataloaders(
            train_dir: str,
            test_dir: str,
            transform: transforms.Compose,
            batch_size: int,
            num_workers: int = NUM_WORKERS
    ):

        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Get class names
        class_names = train_data.classes

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, class_names

    # Setup dataloaders
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                              test_dir=test_dir,
                                                                                              transform=pretrained_vit_transforms,
                                                                                              batch_size=32)
    from going_modular.going_modular import engine

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Addition overfitting countermeasures
    max_norm = 1.0
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the classifier head of the pretrained ViT feature extractor model
    set_seeds()
    results = engine.train(model=pretrained_vit,
                           train_dataloader=train_dataloader_pretrained,
                           test_dataloader=test_dataloader_pretrained,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=25,
                           device=device,
                           max_norm=max_norm,
                           scheduler=scheduler)

    # Metrics calculations and reports
    def calculate_metrics(model, dataloader, device):
        model.eval()
        all_preds = []
        all_labels = []
        correct_preds_per_class = [0] * len(class_names)
        total_per_class = [0] * len(class_names)
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                for i in range(len(labels)):
                    total_per_class[labels[i]] += 1
                    if preds[i] == labels[i]:
                        correct_preds_per_class[labels[i]] += 1

        total_correct = sum(correct_preds_per_class)
        total_samples = sum(total_per_class)
        overall_accuracy = total_correct / total_samples

        return all_preds, all_labels, correct_preds_per_class, total_per_class, overall_accuracy


    def plot_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


    # Calculate metrics after training
    train_preds, train_labels, train_correct, train_total, train_overall_acc = calculate_metrics(pretrained_vit,
                                                                              train_dataloader_pretrained, device)
    test_preds, test_labels, test_correct, test_total, test_overall_acc = calculate_metrics(pretrained_vit, test_dataloader_pretrained,
                                                                          device)

    # Print classification report for train and test
    print("Train Classification Report:")
    print(classification_report(train_labels, train_preds, target_names=class_names))
    print("Test Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Print accuracy per class for train and test
    print("Train Accuracy per Class:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {train_correct[i] / train_total[i] * 100:.2f}%")
    print("Test Accuracy per Class:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {test_correct[i] / test_total[i] * 100:.2f}%")

    print(f"Overall Train Accuracy: {train_overall_acc:.4f}")
    print(f"Overall Test Accuracy: {test_overall_acc:.4f}")


    # Plot confusion matrix for train and test
    print("Train Confusion Matrix:")
    plot_confusion_matrix(train_labels, train_preds, class_names)
    print("Test Confusion Matrix:")
    plot_confusion_matrix(test_labels, test_preds, class_names)


    # Plot the loss curves
    from helper_functions import plot_loss_curves

    plot_loss_curves(results)







