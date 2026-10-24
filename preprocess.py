import os
import shutil
import random
from PIL import Image,ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

SOURCE_DIR="Data/source"
OUTPUT_DIR="Data/preprocessed"
IMAGE_SIZE=(224,224)
AUGMENTATIONS_PER_IMAGE=2
SEED=42
random.seed(SEED)

PETAL_AUGMENT=transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20,interpolation=InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.15,contrast=0.15,saturation=0.15,hue=0.03)
])

def preprocess_image(path):
    with Image.open(path) as image:
        image=ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image,IMAGE_SIZE,method=Image.Resampling.BILINEAR)

def save_image(image,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    image.save(path,format="JPEG",quality=95)

def process_dataset(name,augment=False):
    source=os.path.join(SOURCE_DIR,name)
    output=os.path.join(OUTPUT_DIR,name)
    if not os.path.isdir(source):
        raise FileNotFoundError(f"Dataset not found: {source}")
    if os.path.exists(output):
        shutil.rmtree(output)
    extensions=(".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
    for class_name in sorted(os.listdir(source)):
        class_dir=os.path.join(source,class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith(extensions):
                continue
            path=os.path.join(class_dir,filename)
            try:
                image=preprocess_image(path)
                stem=os.path.splitext(filename)[0]
                save_image(image,os.path.join(output,class_name,f"{stem}.jpg"))
                if augment:
                    for index in range(AUGMENTATIONS_PER_IMAGE):
                        augmented=PETAL_AUGMENT(image.copy())
                        save_image(augmented,os.path.join(output,class_name,f"{stem}_aug{index+1}.jpg"))
            except Exception as error:
                print(f"Skipped {path}: {error}")
    if not any(os.path.isdir(os.path.join(output,c)) and os.listdir(os.path.join(output,c)) for c in os.listdir(output)):
        raise ValueError(f"No valid class images found in {source}")
    print(f"Processed {name}: {output}")

def main():
    process_dataset("rose_petal",augment=True)
    process_dataset("rose_leaf",augment=False)

if __name__=="__main__":
    main()