import os
import sys
from pathlib import Path
import cv2

# Add backend directory to path
backend_dir = Path("c:/Users/gabri/Documents/Projects/PhoSca/backend")
sys.path.append(str(backend_dir))

from app.pipeline.image_ops import load_image, save_jpeg
from app.pipeline.dl_enhancement import dl_enhance_photo
from app.pipeline.diffusion_restoration import premium_enhance_photo
from app.pipeline.segmentation import detect_photos_classical
from app.pipeline.perspective import crop_and_correct_photo

def test_on_photo():
    photo_path = Path("c:/Users/gabri/Documents/Projects/PhoSca/PHOTOALBUM/PXL_20260620_171120181.jpg")
    output_dir = Path("c:/Users/gabri/Documents/Projects/PhoSca/test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading image from {photo_path}")
    image = load_image(photo_path)
    
    print("Detecting photos (classical fallback)...")
    segmentation_result = detect_photos_classical(image)
    print(f"Found {len(segmentation_result.detections)} photos on page.")
    
    for idx, detection in enumerate(segmentation_result.detections):
        print(f"\nProcessing photo #{idx + 1}...")
        
        # 1. Cut the picture out
        cropped_image = crop_and_correct_photo(image, detection.bounding_box, detection.mask)
        
        cropped_path = output_dir / f"test_photo_{idx}_original.jpg"
        save_jpeg(cropped_path, cropped_image)
        print(f"Cropped original saved to {cropped_path}")
        
        # 2. Run standard Deep Learning enhancement
        print("Running standard Deep Learning enhancement...")
        enhanced, metadata = dl_enhance_photo(cropped_image)
        
        enhanced_path = output_dir / f"test_photo_{idx}_enhanced.jpg"
        save_jpeg(enhanced_path, enhanced)
        print(f"Standard enhancement complete. Saved to {enhanced_path}")
        
        # 3. Run premium diffusion enhancement
        print("Running premium diffusion enhancement...")
        premium, premium_metadata = premium_enhance_photo(enhanced)
        premium_path = output_dir / f"test_photo_{idx}_premium.jpg"
        save_jpeg(premium_path, premium)
        print(f"Premium enhancement complete. Saved to {premium_path}")

if __name__ == "__main__":
    test_on_photo()
