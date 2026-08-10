import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Environment Setup
# ==========================================
base_path = '/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/StyleClassificationIndoors/StyleClassificationIndoors'
TRAIN_DIR = os.path.join(base_path, 'train')
TEST_DIR = os.path.join(base_path, 'test')
SAMPLE_SUB_PATH = '/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/sample_submission.csv'
OUTPUT_DIR = './' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Deploying DINOv2-Large | FROZEN LINEAR PROBE | Device: {device}")

# ==========================================
# 2. DataFrame Construction
# ==========================================
classes = sorted(os.listdir(TRAIN_DIR))
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

filepaths, labels = [], []
for cls_name in classes:
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    if os.path.isdir(cls_dir):
        for img_name in os.listdir(cls_dir):
            filepaths.append(os.path.join(cls_dir, img_name))
            labels.append(class_to_idx[cls_name])

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# Penalty Matrix to punish the model for defaulting to Class 10
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1.0 / (class_counts + 1e-6) 
class_weights = class_weights / class_weights.sum() * 17.0 
weight_tensor = torch.FloatTensor(class_weights).to(device)

# ==========================================
# 3. DINOv2 Dataset Engine
# ==========================================
MODEL_ID = "facebook/dinov2-large"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)

class DinoDataset(Dataset):
    def __init__(self, df, processor, is_train=True):
        self.df = df
        self.processor = processor
        self.is_train = is_train
        
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        
        if self.is_train:
            image = self.aug(image)
            
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) 
        label = torch.tensor(row['label'], dtype=torch.long)
        return pixel_values, label

train_dataset = DinoDataset(train_df, processor, is_train=True)
val_dataset = DinoDataset(val_df, processor, is_train=False)

# Because the backbone is frozen, we can use a massive batch size to train at warp speed
BATCH_SIZE = 32 
EPOCHS = 10 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 4. Architecture: 100% Frozen Backbone + Custom MLP
# ==========================================
class DinoClassifier(nn.Module):
    def __init__(self, model_id, num_classes=17):
        super().__init__()
        print(f"\nSecuring {model_id}...")
        self.backbone = AutoModel.from_pretrained(model_id)
        
        # THE LOCKDOWN: 100% of the DINOv2 brain is frozen. 0% chance of overfitting the priors.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # DINOv2-Large outputs a 1024-dimensional geometry vector. 
        # We attach a trainable multi-layer head to interpret it.
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4), # Heavy dropout to prevent the head from memorizing
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # We extract the "CLS Token" which contains the summary of the entire image's geometry
        cls_token = outputs.last_hidden_state[:, 0, :] 
        logits = self.head(cls_token)
        return logits

model = DinoClassifier(MODEL_ID, num_classes=17).to(device)

# ==========================================
# 5. The Probe Engine
# ==========================================
# We ONLY pass the `head` parameters to the optimizer. The backbone uses zero VRAM for gradients.
optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

total_steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

best_val_acc = 0.0

print(f"\n[ENGINE START] Initiating {EPOCHS}-Epoch Frozen Probe...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0 
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1) 
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = running_corrects.double() / len(train_dataset) 
    
    # --- VALIDATION ---
    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_preds, all_labels = [], [] 
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels) 
                _, preds = torch.max(outputs, 1)
                
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_dino_model.pth")

# ==========================================
# 6. Advanced Inference (TTA)
# ==========================================
print(f"\n[INFERENCE] Loading Best Weights (Val Acc: {best_val_acc:.4f})...")
model.load_state_dict(torch.load("best_dino_model.pth"))
model.eval()

submission_df = pd.read_csv(SAMPLE_SUB_PATH)
hard_predictions = []

with torch.no_grad():
    for index, row in submission_df.iterrows():
        try:
            img_path = os.path.join(TEST_DIR, row['ImageName'])
            img = Image.open(img_path).convert('RGB')
            
            t1 = processor(images=img, return_tensors="pt")['pixel_values'].to(device)
            t2 = processor(images=img.transpose(Image.FLIP_LEFT_RIGHT), return_tensors="pt")['pixel_values'].to(device)
            
            with torch.amp.autocast('cuda'):
                p1 = F.softmax(model(t1), dim=1)
                p2 = F.softmax(model(t2), dim=1)
                
            probabilities = (p1 * 0.60 + p2 * 0.40).squeeze().cpu().numpy()
            pred_idx = int(np.argmax(probabilities))
            hard_predictions.append(pred_idx)
            
        except Exception:
            hard_predictions.append(0) 

final_df = pd.DataFrame({'ImageName': submission_df['ImageName'], 'ClassLabel': hard_predictions})

csv_path = os.path.join(OUTPUT_DIR, 'final_submission_DINOv2.csv')
final_df.to_csv(csv_path, index=False)
print(f"\nSUCCESS! DINOv2 Predictions secured to: {csv_path}")
