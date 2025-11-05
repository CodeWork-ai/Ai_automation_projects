# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# from PIL import Image
# import time
# from tqdm import tqdm
# import json
# from datetime import datetime

# # Configuration
# DATA_DIR = r"D:\fire\extracted_frames"
# MODELS_DIR = os.path.join(DATA_DIR, "models")
# RESULTS_DIR = os.path.join(DATA_DIR, "results")
# IMG_SIZE = 64
# BATCH_SIZE = 64
# EPOCHS = 25
# TARGET_TOTAL_IMAGES = 10000

# # Create necessary directories
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# print("=" * 60)
# print("BALANCED CNN IMAGE CLASSIFIER (PyTorch)")
# print("=" * 60)
# print(f"Target dataset size: {TARGET_TOTAL_IMAGES:,} images")


# # Step 1: Analyze dataset and calculate balancing
# def analyze_dataset(data_dir):
#     """Count images in each class and calculate balancing strategy"""
#     class_info = {}
#     class_folders = sorted([f for f in os.listdir(data_dir)
#                             if os.path.isdir(os.path.join(data_dir, f))
#                             and not f.startswith('.')])

#     # First pass: count images and filter out empty folders
#     valid_class_folders = []
#     print(f"\nScanning folders...")
#     for class_name in class_folders:
#         class_path = os.path.join(data_dir, class_name)
#         image_files = [f for f in os.listdir(class_path)
#                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

#         if len(image_files) > 0:  # Only include folders with images
#             valid_class_folders.append(class_name)
#             class_info[class_name] = {
#                 'path': class_path,
#                 'files': image_files,
#                 'count': len(image_files)
#             }
#             print(f"  {class_name}: {len(image_files):,} images")
#         else:
#             print(f"  {class_name}: 0 images (skipping)")

#     class_folders = valid_class_folders
#     print(f"\nFound {len(class_folders)} valid classes with images")

#     # Calculate target per class
#     num_classes = len(class_folders)
#     target_per_class = TARGET_TOTAL_IMAGES // num_classes

#     print(f"\nBalancing strategy:")
#     print(f"  Target per class: {target_per_class:,} images")

#     for class_name, info in class_info.items():
#         original_count = info['count']
#         if original_count >= target_per_class:
#             info['strategy'] = 'undersample'
#             info['target_count'] = target_per_class
#             info['copies_per_image'] = 1
#             info['images_to_use'] = target_per_class
#         else:
#             info['strategy'] = 'oversample'
#             info['target_count'] = target_per_class
#             info['copies_per_image'] = target_per_class // original_count
#             info['images_to_use'] = original_count

#         print(f"  {class_name}: {info['strategy']} "
#               f"({original_count:,} → {info['target_count']:,})")

#     return class_info, class_folders


# # Custom dataset with smart balancing
# class BalancedImageDataset(Dataset):
#     def __init__(self, data_dir, transform=None, target_total=10000):
#         self.data = []
#         self.labels = []
#         self.class_names = []
#         self.transform = transform

#         # Analyze and balance
#         self.class_info, self.class_names = analyze_dataset(data_dir)

#         print(f"\nBuilding balanced dataset...")
#         for class_idx, class_name in enumerate(self.class_names):
#             info = self.class_info[class_name]

#             if info['strategy'] == 'undersample':
#                 # Randomly sample subset of images
#                 selected_files = np.random.choice(
#                     info['files'],
#                     size=info['images_to_use'],
#                     replace=False
#                 )
#                 for img_file in selected_files:
#                     img_path = os.path.join(info['path'], img_file)
#                     self.data.append(img_path)
#                     self.labels.append(class_idx)
#             else:
#                 # Oversample by creating multiple copies
#                 for img_file in info['files']:
#                     img_path = os.path.join(info['path'], img_file)
#                     for _ in range(info['copies_per_image']):
#                         self.data.append(img_path)
#                         self.labels.append(class_idx)

#                 # Add remaining images to reach target
#                 remaining = info['target_count'] - (info['images_to_use'] * info['copies_per_image'])
#                 if remaining > 0:
#                     extra_files = np.random.choice(info['files'], size=remaining, replace=True)
#                     for img_file in extra_files:
#                         img_path = os.path.join(info['path'], img_file)
#                         self.data.append(img_path)
#                         self.labels.append(class_idx)

#         print(f"\nFinal dataset: {len(self.data):,} images")

#         # Verify balance
#         unique, counts = np.unique(self.labels, return_counts=True)
#         print("\nClass distribution:")
#         for class_idx, count in zip(unique, counts):
#             print(f"  {self.class_names[class_idx]}: {count:,} images")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = self.data[idx]
#         label = self.labels[idx]

#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# # Data transforms
# train_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Step 2: Load dataset
# print("\n" + "=" * 60)
# print("[1/4] Loading and balancing dataset...")
# print("=" * 60)
# start_time = time.time()

# full_dataset = BalancedImageDataset(DATA_DIR, transform=train_transform, target_total=TARGET_TOTAL_IMAGES)
# num_classes = len(full_dataset.class_names)

# # Split: 80% train, 20% test
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# # Apply test transform to test dataset
# test_dataset.dataset.transform = test_transform

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# print(f"\nTrain: {len(train_dataset):,} | Test: {len(test_dataset):,}")
# print(f"Loading time: {time.time() - start_time:.2f}s")

# # Step 3: Define CNN model
# print("\n" + "=" * 60)
# print("[2/4] Building model...")
# print("=" * 60)


# class FireCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(FireCNN, self).__init__()

#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.2),

#             # Block 2
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.2),

#             # Block 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.3),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 8 * 8, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# model = FireCNN(num_classes=num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# # Step 4: Training
# print("\n" + "=" * 60)
# print("[3/4] Training model...")
# print("=" * 60)
# train_start = time.time()

# training_history = {
#     'train_acc': [],
#     'train_loss': [],
#     'val_acc': [],
#     'val_loss': []
# }

# best_acc = 0.0
# patience_counter = 0
# patience_limit = 5

# for epoch in range(EPOCHS):
#     # Training phase
#     model.train()
#     train_loss = 0.0
#     train_correct = 0
#     train_total = 0

#     pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
#     for images, labels in pbar:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         train_total += labels.size(0)
#         train_correct += (predicted == labels).sum().item()

#         pbar.set_postfix({'loss': f'{loss.item():.4f}',
#                           'acc': f'{100 * train_correct / train_total:.2f}%'})

#     train_acc = 100 * train_correct / train_total
#     avg_train_loss = train_loss / len(train_loader)

#     # Validation phase
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0

#     with torch.no_grad():
#         for images, labels in test_loader:
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             val_total += labels.size(0)
#             val_correct += (predicted == labels).sum().item()

#     val_acc = 100 * val_correct / val_total
#     avg_val_loss = val_loss / len(test_loader)

#     scheduler.step(avg_val_loss)

#     # Store history
#     training_history['train_acc'].append(train_acc)
#     training_history['train_loss'].append(avg_train_loss)
#     training_history['val_acc'].append(val_acc)
#     training_history['val_loss'].append(avg_val_loss)

#     print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
#           f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#     # Early stopping
#     if val_acc > best_acc:
#         best_acc = val_acc
#         torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pth'))
#         patience_counter = 0
#     else:
#         patience_counter += 1
#         if patience_counter >= patience_limit:
#             print(f"\nEarly stopping triggered after {epoch + 1} epochs")
#             break

# train_time = time.time() - train_start
# print(f"\nTraining completed in {train_time:.2f}s ({train_time / 60:.2f} minutes)")

# # Step 5: Final evaluation
# print("\n" + "=" * 60)
# print("[4/4] Final Evaluation...")
# print("=" * 60)
# model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model.pth')))
# model.eval()

# # Detailed evaluation
# test_correct = 0
# test_total = 0
# class_correct = [0] * num_classes
# class_total = [0] * num_classes

# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         test_total += labels.size(0)
#         test_correct += (predicted == labels).sum().item()

#         # Per-class accuracy
#         for label, pred in zip(labels, predicted):
#             class_total[label] += 1
#             if label == pred:
#                 class_correct[label] += 1

# test_acc = 100 * test_correct / test_total

# # Per-class accuracy
# class_accuracies = {}
# for i in range(num_classes):
#     if class_total[i] > 0:
#         class_acc = 100 * class_correct[i] / class_total[i]
#         class_accuracies[full_dataset.class_names[i]] = class_acc

# # Save results
# results = {
#     'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     'configuration': {
#         'data_dir': DATA_DIR,
#         'target_images': TARGET_TOTAL_IMAGES,
#         'img_size': IMG_SIZE,
#         'batch_size': BATCH_SIZE,
#         'epochs': epoch + 1,
#         'model_params': sum(p.numel() for p in model.parameters())
#     },
#     'dataset': {
#         'num_classes': num_classes,
#         'class_names': full_dataset.class_names,
#         'train_size': len(train_dataset),
#         'test_size': len(test_dataset),
#         'total_size': len(full_dataset)
#     },
#     'performance': {
#         'best_val_accuracy': float(best_acc),
#         'final_test_accuracy': float(test_acc),
#         'class_accuracies': {k: float(v) for k, v in class_accuracies.items()}
#     },
#     'training_time': {
#         'total_seconds': float(train_time),
#         'minutes': float(train_time / 60)
#     },
#     'training_history': {
#         'train_acc': [float(x) for x in training_history['train_acc']],
#         'train_loss': [float(x) for x in training_history['train_loss']],
#         'val_acc': [float(x) for x in training_history['val_acc']],
#         'val_loss': [float(x) for x in training_history['val_loss']]
#     },
#     'model_path': os.path.join(MODELS_DIR, 'best_model.pth'),
#     'class_mapping': dict(enumerate(full_dataset.class_names))
# }

# # Save JSON results
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# results_file = os.path.join(RESULTS_DIR, f'training_results_{timestamp}.json')
# with open(results_file, 'w') as f:
#     json.dump(results, f, indent=2)

# # Save simple text summary
# summary_file = os.path.join(RESULTS_DIR, f'summary_{timestamp}.txt')
# with open(summary_file, 'w') as f:
#     f.write("=" * 60 + "\n")
#     f.write("FIRE CLASSIFIER TRAINING RESULTS\n")
#     f.write("=" * 60 + "\n\n")
#     f.write(f"Training Date: {results['timestamp']}\n\n")

#     f.write("DATASET:\n")
#     f.write(f"  Total Images: {len(full_dataset):,}\n")
#     f.write(f"  Classes: {num_classes}\n")
#     for name in full_dataset.class_names:
#         f.write(f"    - {name}\n")
#     f.write(f"  Train/Test Split: {len(train_dataset):,} / {len(test_dataset):,}\n\n")

#     f.write("MODEL:\n")
#     f.write(f"  Architecture: FireCNN\n")
#     f.write(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")

#     f.write("PERFORMANCE:\n")
#     f.write(f"  Best Validation Accuracy: {best_acc:.2f}%\n")
#     f.write(f"  Final Test Accuracy: {test_acc:.2f}%\n\n")

#     f.write("PER-CLASS ACCURACY:\n")
#     for class_name, acc in class_accuracies.items():
#         f.write(f"  {class_name}: {acc:.2f}%\n")

#     f.write(f"\nTraining Time: {train_time / 60:.2f} minutes\n")
#     f.write(f"\nModel saved to: {os.path.join(MODELS_DIR, 'best_model.pth')}\n")

# # Print final results
# print("\n" + "=" * 60)
# print("FINAL RESULTS")
# print("=" * 60)
# print(f"Best Validation Accuracy: {best_acc:.2f}%")
# print(f"Final Test Accuracy: {test_acc:.2f}%")
# print(f"\nPer-Class Accuracy:")
# for class_name, acc in class_accuracies.items():
#     print(f"  {class_name}: {acc:.2f}%")
# print(f"\nTotal Time: {time.time() - start_time:.2f}s ({(time.time() - start_time) / 60:.2f} min)")
# print(f"\nModel saved to: {os.path.join(MODELS_DIR, 'best_model.pth')}")
# print(f"Results saved to: {results_file}")
# print(f"Summary saved to: {summary_file}")

# print("\n✓ Training complete! Model and results saved.")
