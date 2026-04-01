import os
import random
from PIL import Image, ImageFilter, ImageEnhance
import io
import numpy as np

def augment_document(image_path, output_dir, count=30):
    """Generate `count` augmented versions of a document."""
    img = Image.open(image_path).convert("RGB")
    os.makedirs(output_dir, exist_ok=True)
    
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    for i in range(count):
        aug = img.copy()
        
        # 1. Mild rotation (scanned docs are slightly tilted)
        angle = random.uniform(-3, 3)
        aug = aug.rotate(angle, fillcolor=(255, 255, 255))
        
        # 2. Random brightness
        aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.80, 1.20))
        
        # 3. Random contrast
        aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.80, 1.20))
        
        # 4. Random sharpness
        aug = ImageEnhance.Sharpness(aug).enhance(random.uniform(0.5, 2.0))
        
        # 5. Random color saturation
        aug = ImageEnhance.Color(aug).enhance(random.uniform(0.8, 1.2))
        
        # 6. Random blur (simulates bad scan)
        if random.random() < 0.3:
            aug = aug.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.5, 1.5)
            ))
        
        # 7. Random crop + resize (simulates different scan borders)
        if random.random() < 0.4:
            w, h = aug.size
            left   = random.randint(0, int(w * 0.05))
            top    = random.randint(0, int(h * 0.05))
            right  = random.randint(int(w * 0.95), w)
            bottom = random.randint(int(h * 0.95), h)
            aug = aug.crop((left, top, right, bottom))
            aug = aug.resize((w, h), Image.LANCZOS)
        
        # 8. Random horizontal flip (some scanners mirror)
        if random.random() < 0.1:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 9. Add slight noise (simulates scanner grain)
        if random.random() < 0.4:
            arr = np.array(aug).astype(np.int16)
            noise = np.random.randint(-15, 15, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            aug = Image.fromarray(arr)
        
        # 10. Random JPEG quality — CRITICAL for ELA variety
        quality = random.randint(70, 95)
        buf = io.BytesIO()
        aug.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        aug = Image.open(buf).copy()
        
        out_path = os.path.join(output_dir, f"{base}_aug{i:03d}.jpg")
        aug.save(out_path, "JPEG", quality=85)
    
    print(f"✅ {os.path.basename(image_path)} → {count} augmented images")


# ─── Config ────────────────────────────────────────────
DATASET_DIR = r"D:\Final year project\DocumentAnalyser-main\DocumentAnalyser-main\document_fraud_ai\data_dir"
AUGMENT_COUNT = 30  # 60 genuine × 30 = 1800 | 60 tampered × 30 = 1800

# ─── Run ───────────────────────────────────────────────
for label in ["genuine", "tampered"]:
    src = os.path.join(DATASET_DIR, label)
    dst = os.path.join(DATASET_DIR, "augmented", label)
    
    files = [f for f in os.listdir(src) 
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    print(f"\n📁 Processing {label}: {len(files)} images × {AUGMENT_COUNT} = {len(files) * AUGMENT_COUNT} total")
    
    for fname in files:
        augment_document(
            image_path=os.path.join(src, fname),
            output_dir=dst,
            count=AUGMENT_COUNT
        )

print("\n🎉 Done! Check dataset/augmented/")

# ─── Summary ───────────────────────────────────────────
for label in ["genuine", "tampered"]:
    dst = os.path.join(DATASET_DIR, "augmented", label)
    count = len(os.listdir(dst))
    print(f"  {label:10}: {count} images")