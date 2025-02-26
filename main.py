from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GAE

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

# Define input schema for prediction
class MachineInput(BaseModel):
    M1_Status: int
    M1_Worker_Count: int
    M2_Status: int
    M2_Worker_Count: int
    M3_Status: int
    M3_Worker_Count: int

app = FastAPI()

# Recreate and load the trained model
in_channels = 3
hidden_channels = 32
out_channels = 16

encoder = GATEncoder(in_channels, hidden_channels, out_channels, heads=8, dropout=0.6)
model = GAE(encoder)
model.load_state_dict(torch.load("gat_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.post("/predict")
def predict(input: MachineInput):

    node_features = [
        [input.M1_Status, input.M1_Worker_Count, 0],
        [input.M2_Status, input.M2_Worker_Count, 0],
        [input.M3_Status, input.M3_Worker_Count, 0]
    ]
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Define sequential edges: M1 -> M2, M2 -> M3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        reconstructed_adj = torch.sigmoid(torch.matmul(z, z.t()))
        original_adj = torch.zeros((3, 3))
        original_adj[0, 1] = 1.0
        original_adj[1, 2] = 1.0
        mse = torch.mean((original_adj - reconstructed_adj) ** 2).item()
    
    anomaly_threshold = 0.75
    is_anomaly = mse > anomaly_threshold
    
    return {"reconstruction_error": mse, "anomaly": is_anomaly}
