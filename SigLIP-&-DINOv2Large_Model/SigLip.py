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
from transformers import AutoProcessor, SiglipVisionModel, get_cosine_schedule_with_warmup
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
print(f"Deploying SigLIP-SO400M (Grandmaster Pipeline) | Device: {device}")

# ==========================================
# 2. DataFrame Construction & Strict Splitting
# ==========================================
print("\nConstructing Strict Stratified Dataset...")
classes = sorted(os.listdir(TRAIN_DIR))
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

filepaths = []
labels = []
for cls_name in classes:
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    if os.path.isdir(cls_dir):
        for img_name in os.listdir(cls_dir):
            filepaths.append(os.path.join(cls_dir, img_name))
            labels.append(class_to_idx[cls_name])

df = pd.DataFrame({'filepath': filepaths, 'label': labels})

# 90/10 Stratified Split ensures every class is perfectly represented in Validation
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# --- THE PENALTY MATRIX ---
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1.0 / (class_counts + 1e-6) 
class_weights = class_weights / class_weights.sum() * 17.0 
weight_tensor = torch.FloatTensor(class_weights).to(device)

# ==========================================
# 3. The Hugging Face Dataset Engine
# ==========================================
MODEL_ID = "google/siglip-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(MODEL_ID)

class SiglipDataset(Dataset):
    def __init__(self, df, processor, is_train=True):
        self.df = df
        self.processor = processor
        self.is_train = is_train
        
        # We apply PIL-level augmentations BEFORE the Hugging Face processor
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        
        if self.is_train:
            image = self.aug(image)
            
        # The processor handles the exact resizing, center cropping, and normalization SigLIP expects
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) 
        
        label = torch.tensor(row['label'], dtype=torch.long)
        return pixel_values, label

train_dataset = SiglipDataset(train_df, processor, is_train=True)
val_dataset = SiglipDataset(val_df, processor, is_train=False)

# SO400M is massive. Tiny batch size, high accumulation.
BATCH_SIZE = 4 
ACCUMULATION_STEPS = 16 
EPOCHS = 10  # We have infinite time, expanding to 10 epochs for full convergence

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 4. Architecture: SigLIP SO400M Custom Wrapper
# ==========================================
class SiglipClassifier(nn.Module):
    def __init__(self, model_id, num_classes=17):
        super().__init__()
        print(f"\nDownloading and injecting {model_id} Vision Backbone...")
        self.vision_model = SiglipVisionModel.from_pretrained(model_id)
        
        # FREEZE PROTOCOL: Freeze the first 60% of the model to protect deep visual priors
        num_layers = len(self.vision_model.vision_model.encoder.layers)
        freeze_layers = int(num_layers * 0.6)
        
        self.vision_model.vision_model.embeddings.requires_grad_(False)
        for i in range(freeze_layers):
            for param in self.vision_model.vision_model.encoder.layers[i].parameters():
                param.requires_grad = False
                
        # The Custom Classification Head
        hidden_size = self.vision_model.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        # SigLIP natively uses a Multihead Attention pooling output
        pooled_output = outputs.pooler_output 
        logits = self.head(pooled_output)
        return logits

model = SiglipClassifier(MODEL_ID, num_classes=17).to(device)

# ==========================================
# 5. The Grandmaster Training Loop
# ==========================================
# We use a very conservative learning rate because the backbone is pre-trained
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=0.1)
criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.15)
scaler = torch.amp.GradScaler('cuda')

# Cosine Schedule with Warmup is mandatory for fine-tuning Foundation Models
total_steps = math.ceil(len(train_loader) / ACCUMULATION_STEPS) * EPOCHS
warmup_steps = int(total_steps * 0.1) # 10% of training time spent warming up
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

best_val_acc = 0.0

print(f"\n[ENGINE START] Initiating {EPOCHS}-Epoch SO400M Deep Fine-Tune...")
print(f"Total Steps: {total_steps} | Warmup Steps: {warmup_steps}\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0 
    optimizer.zero_grad()
    
    for i, (pixel_values, labels) in enumerate(train_loader):
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values)
            loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            _, preds = torch.max(outputs, 1) 
            
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
        running_loss += loss.item() * ACCUMULATION_STEPS * pixel_values.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = running_corrects.double() / len(train_dataset) 
    
    # --- VALIDATION ---
    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_preds, all_labels = [], [] 
    
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values)
                loss = criterion(outputs, labels) 
                _, preds = torch.max(outputs, 1)
                
            val_loss += loss.item() * pixel_values.size(0)
            val_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e} | Train Loss: {epoch_train_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
    # Save the absolute best weights
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_siglip_model.pth")
        print("   -> 🌟 New Best Model Saved!")

# ==========================================
# 6. Advanced Inference (TTA + Best Weights)
# ==========================================
print(f"\n[INFERENCE] Loading Best Weights (Val Acc: {best_val_acc:.4f}) and running TTA...")
model.load_state_dict(torch.load("best_siglip_model.pth"))
model.eval()

submission_df = pd.read_csv(SAMPLE_SUB_PATH)
hard_predictions = []

with torch.no_grad():
    for index, row in submission_df.iterrows():
        try:
            img_path = os.path.join(TEST_DIR, row['ImageName'])
            img = Image.open(img_path).convert('RGB')
            
            # Pass 1: Original Image
            t1 = processor(images=img, return_tensors="pt")['pixel_values'].to(device)
            
            # Pass 2: Horizontally Flipped Image
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            t2 = processor(images=img_flip, return_tensors="pt")['pixel_values'].to(device)
            
            with torch.amp.autocast('cuda'):
                p1 = F.softmax(model(t1), dim=1)
                p2 = F.softmax(model(t2), dim=1)
                
            # Average the predictions
            probabilities = (p1 * 0.55 + p2 * 0.45).squeeze().cpu().numpy()
            pred_idx = int(np.argmax(probabilities))
            
            hard_predictions.append(pred_idx)
            
        except Exception:
            hard_predictions.append(0) 

final_df = pd.DataFrame({'ImageName': submission_df['ImageName'], 'ClassLabel': hard_predictions})

print("\n=== Final Test Set Class Distribution ===")
dist = final_df['ClassLabel'].value_counts().sort_index()
for cls_lbl, count in dist.items(): print(f"Class {cls_lbl}: {count} predictions")
print("=========================================")

csv_path = os.path.join(OUTPUT_DIR, 'final_submission_SIGLIP_SO400M.csv')
final_df.to_csv(csv_path, index=False)
print(f"\nSUCCESS! Ultimate Predictions secured to: {csv_path}")
