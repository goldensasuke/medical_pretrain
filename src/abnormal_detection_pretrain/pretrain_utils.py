import torch
import numpy as np
import wandb
from .models import get_model, MaskedAutoencoder

def pretrain(data_npy_path, model_name='efficientnet_b0', epochs=5, lr=1e-4, project_name='pretrain_mae'):
    data = np.load(data_npy_path)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    model = get_model(model_name)
    mae = MaskedAutoencoder(model)
    
    optimizer = torch.optim.Adam(mae.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    wandb.init(project=project_name, config={"epochs": epochs, "lr": lr, "model_name": model_name})
    
    for epoch in range(epochs):
        mae.train()
        optimizer.zero_grad()
        output = mae(data_tensor)
        loss = criterion(output, data_tensor)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        wandb.log({"epoch": epoch+1, "loss": loss.item()})
    
    model_path = os.path.join(output_dir, f"{model_name}_pretrained_mae.pth")
    torch.save(mae.state_dict(), model_path)
    print(f"Pretraining done, model saved to {model_path}")
    wandb.finish()
