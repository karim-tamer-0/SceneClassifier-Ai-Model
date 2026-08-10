import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Environment Setup
# ==========================================
base_path = '/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/StyleClassificationIndoors/StyleClassificationIndoors'
TEST_DIR = os.path.join(base_path, 'test')
SAMPLE_SUB_PATH = '/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/sample_submission.csv'
OUTPUT_DIR = './' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Deploying 5-CROP TTA ASSAULT | Alpha Engine: SigLIP-SO400M | Device: {device}")

# ==========================================
# 2. Rebuild the Alpha Architecture
# ==========================================
MODEL_ID = "google/siglip-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(MODEL_ID)

class SiglipClassifier(nn.Module):
    def __init__(self, model_id, num_classes=17):
        super().__init__()
        self.vision_model = SiglipVisionModel.from_pretrained(model_id)
        hidden_size = self.vision_model.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output 
        logits = self.head(pooled_output)
        return logits

model = SiglipClassifier(MODEL_ID, num_classes=17).to(device)

# LOAD YOUR SAVED WEIGHTS HERE
print("Loading best_siglip_model.pth...")
model.load_state_dict(torch.load("best_siglip_model.pth", map_location=device))
model.eval()

# ==========================================
# 3. The 5-Crop Augmentation Pipeline
# ==========================================
# 1. Original Image (Handled dynamically)
# 2. Horizontal Flip
flip = transforms.RandomHorizontalFlip(p=1.0)
# 3. 110% Zoom & Crop (Removes borders/edges)
zoom_crop = transforms.Compose([
    transforms.Resize((420, 420)),
    transforms.CenterCrop(384)
])
# 4. Brightened (+20%)
brighten = transforms.ColorJitter(brightness=(1.2, 1.2))
# 5. Darkened (-20%)
darken = transforms.ColorJitter(brightness=(0.8, 0.8))

print("\n[INFERENCE] Commencing 5-View Interrogation...")

submission_df = pd.read_csv(SAMPLE_SUB_PATH)
hard_predictions = []

with torch.no_grad():
    for index, row in submission_df.iterrows():
        try:
            img_path = os.path.join(TEST_DIR, row['ImageName'])
            img = Image.open(img_path).convert('RGB')
            
            # Generate the 5 views
            views = [
                img,                     # 1. Original
                flip(img),               # 2. Flipped
                zoom_crop(img),          # 3. Zoomed in
                brighten(img),           # 4. Brightened
                darken(img)              # 5. Darkened
            ]
            
            # Process all 5 views
            probs_list = []
            for view in views:
                t_tensor = processor(images=view, return_tensors="pt")['pixel_values'].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(t_tensor)
                    probs = F.softmax(logits, dim=1)
                    probs_list.append(probs)
            
            # Mathematical Average of all 5 views
            stacked_probs = torch.stack(probs_list)
            avg_probs = torch.mean(stacked_probs, dim=0).squeeze().cpu().numpy()
            
            pred_idx = int(np.argmax(avg_probs))
            hard_predictions.append(pred_idx)
            
        except Exception:
            hard_predictions.append(0) 

final_df = pd.DataFrame({'ImageName': submission_df['ImageName'], 'ClassLabel': hard_predictions})

print("\n=== 5-Crop Test Set Class Distribution ===")
dist = final_df['ClassLabel'].value_counts().sort_index()
for cls_lbl, count in dist.items(): print(f"Class {cls_lbl}: {count} predictions")
print("=========================================")

csv_path = os.path.join(OUTPUT_DIR, 'final_submission_SIGLIP_5CROP.csv')
final_df.to_csv(csv_path, index=False)
print(f"\nSUCCESS! 5-Crop Assault complete. Submit: {csv_path}")
