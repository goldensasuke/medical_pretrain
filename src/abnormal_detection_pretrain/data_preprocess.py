import os
import glob
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_images(data_dir, output_dir, image_size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    img_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
        img_files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    npy_list = []
    metadata = []
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    for img_path in img_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        npy_list.append(img_tensor.numpy())
        metadata.append({'file': os.path.basename(img_path)})
    
    npy_array = np.stack(npy_list)
    np.save(os.path.join(output_dir, 'images.npy'), npy_array)
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved {len(img_files)} images to {output_dir}")
