import torch
import numpy as np
import wandb
import os
from .models import get_model, MaskedAutoencoder

def pretrain(data_npy_path, model_name='efficientnet_b0', epochs=5, lr=1e-4, batch_size=16, project_name='pretrain_mae', output_dir='.'):
    # 載入資料
    data = np.load(data_npy_path)
    dataset_size = len(data)
    
    model = get_model(model_name)
    mae = MaskedAutoencoder(model)
    
    optimizer = torch.optim.Adam(mae.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    wandb.init(project=project_name, config={"epochs": epochs, "lr": lr, "model_name": model_name, "batch_size": batch_size})
    
    # 將資料轉成 Tensor（可分批處理）
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    for epoch in range(epochs):
        mae.train()
        epoch_loss = 0.0
        
        # 分批處理
        for i in range(0, dataset_size, batch_size):
            batch = data_tensor[i:i+batch_size]
            optimizer.zero_grad()
            output = mae(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        
        avg_loss = epoch_loss / dataset_size
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "loss": avg_loss})
    
    # 儲存模型
    model_path = os.path.join(output_dir, f"{model_name}_pretrained_mae.pth")
    torch.save(mae.state_dict(), model_path)
    print(f"Pretraining done, model saved to {model_path}")
    wandb.finish()
