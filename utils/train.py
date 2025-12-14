import os
import torch
import torch_geometric as pyg
from model import LearnedSimulator
from dataset import OneStepDataset, RolloutDataset

# Params
params = {
    "noise": 3e-4,
    "batch_size": 2,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 10,  # Adjusted for demo
    "model_dir": "temp/models/WaterDrop"
}

def train(params, simulator, train_loader, valid_loader):
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    loss_fn = torch.nn.MSELoss()
    
    if not os.path.exists(params["model_dir"]):
        os.makedirs(params["model_dir"])

    simulator.train()
    step = 0
    
    for epoch in range(params["epochs"]):
        print(f"Epoch {epoch+1}")
        for batch in train_loader:
            batch = batch.to(params["device"])
            optimizer.zero_grad()
            
            pred_acc = simulator(batch)
            loss = loss_fn(pred_acc, batch.y)
            
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")
                
        # Save checkpoint
        torch.save(simulator.state_dict(), os.path.join(params["model_dir"], f"checkpoint_{step}.pt"))

if __name__ == "__main__":
    data_path = "temp/datasets/WaterDrop"
    
    train_ds = OneStepDataset(data_path, "train", noise_std=params["noise"])
    valid_ds = OneStepDataset(data_path, "valid", noise_std=0.0)
    
    train_loader = pyg.loader.DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    valid_loader = pyg.loader.DataLoader(valid_ds, batch_size=params["batch_size"], shuffle=False)
    
    simulator = LearnedSimulator().to(params["device"])
    
    train(params, simulator, train_loader, valid_loader)