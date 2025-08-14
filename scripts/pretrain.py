#!/usr/bin/env python
import argparse
from abnormal_detection_pretrain.data_preprocess import preprocess_images
from abnormal_detection_pretrain.pretrain_utils import pretrain
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw image folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save processed .npy/.pkl')
    parser.add_argument('--image_size', type=int, default=224, help='Resize image to square size')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='Backbone model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of pretraining epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--project_name', type=str, default='pretrain_mae', help='WandB project name')
    
    args = parser.parse_args()
    
    preprocess_images(args.data_dir, args.output_dir, image_size=(args.image_size,args.image_size))
    pretrain(os.path.join(args.output_dir, 'images.npy'),
             model_name=args.model_name,
             epochs=args.epochs,
             lr=args.lr,
             project_name=args.project_name)
