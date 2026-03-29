import os
import random
from PIL import Image, ImageFilter
import io

def augment_document(image_path, output_dir, label, count=6):
    """Generate `count` augmented versions of a document."""
    img = Image.open(image_path).convert("RGB")
    os.makedirs(output_dir, exist_ok=True)
    
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    for i in range(count):
        aug = img.copy()
        
        # Mild rotation (scanned docs are slightly tilted)
        angle = random.uniform(-2, 2)
        aug = aug.rotate(angle, fillcolor=(255, 255, 255))
        
        # Random brightness (different scanner settings)
        from PIL import ImageEnhance
        aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.85, 1.15))
        aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.85, 1.15))
        
        # Random JPEG quality — CRITICAL for ELA variety
        quality = random.randint(70, 95)
        buf = io.BytesIO()
        aug.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        aug = Image.open(buf).copy()
        
        out_path = os.path.join(output_dir, f"{base}_aug{i}.jpg")
        aug.save(out_path, "JPEG", quality=85)
        print(f"Saved: {out_path}")


# Run on your dataset
DATASET_DIR = r"D:\Final year project\DocumentAnalyser-main\DocumentAnalyser-main\document_fraud_ai\Dataset"

for label in ["genuine", "tampered"]:
    src = os.path.join(DATASET_DIR, label)
    dst = os.path.join(DATASET_DIR, "augmented", label)
    
    for fname in os.listdir(src):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            augment_document(
                os.path.join(src, fname),
                dst,
                label,
                count=6  # 100 × 6 = 600 total
            )

print("Done! Check dataset/augmented/")