import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GAE
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('manufacturing_test_data.csv')

# Map machine status to binary (Running=1, Stopped=0)
status_map = {"Running": 1, "Stopped": 0}
df["M1_Status"] = df["M1_Status"].map(status_map)
df["M2_Status"] = df["M2_Status"].map(status_map)
df["M3_Status"] = df["M3_Status"].map(status_map)

# Create a directed graph with NetworkX
G = nx.DiGraph()

num_rows = len(df)
for i in range(num_rows):
    # --- Machine Nodes ---
    m1_node = f"M1_{i}"
    m2_node = f"M2_{i}"
    m3_node = f"M3_{i}"
    
    G.add_node(m1_node, status=df.loc[i, "M1_Status"], Worker_Counts=df.loc[i, "M1_Worker_Count"], node_type=0)
    G.add_node(m2_node, status=df.loc[i, "M2_Status"], Worker_Counts=df.loc[i, "M2_Worker_Count"], node_type=0)
    G.add_node(m3_node, status=df.loc[i, "M3_Status"], Worker_Counts=df.loc[i, "M3_Worker_Count"], node_type=0)
    
    # --- Sequential Edges for Machines at the same timestamp ---
    G.add_edge(m1_node, m2_node)
    G.add_edge(m2_node, m3_node)
    
    # --- Worker Nodes & Machine-Worker Edges ---
    # For Machine 1
    for j in range(int(df.loc[i, "M1_Worker_Count"])):
        worker_node = f"Worker_M1_{i}_{j}"
        G.add_node(worker_node, status=0, Worker_Counts=0, node_type=1)
        # Add bidirectional edges
        G.add_edge(m1_node, worker_node)
        G.add_edge(worker_node, m1_node)
        
    # For Machine 2
    for j in range(int(df.loc[i, "M2_Worker_Count"])):
        worker_node = f"Worker_M2_{i}_{j}"
        G.add_node(worker_node, status=0, Worker_Counts=0, node_type=1)
        G.add_edge(m2_node, worker_node)
        G.add_edge(worker_node, m2_node)
        
    # For Machine 3
    for j in range(int(df.loc[i, "M3_Worker_Count"])):
        worker_node = f"Worker_M3_{i}_{j}"
        G.add_node(worker_node, status=0, Worker_Counts=0, node_type=1)
        G.add_edge(m3_node, worker_node)
        G.add_edge(worker_node, m3_node)
        
# --- Temporal Edges for Machine Nodes Across Timestamps ---
for i in range(num_rows - 1):
    G.add_edge(f"M1_{i}", f"M1_{i+1}")
    G.add_edge(f"M2_{i}", f"M2_{i+1}")
    G.add_edge(f"M3_{i}", f"M3_{i+1}")

# Convert the graph to a PyTorch Geometric Data object
node_names = list(G.nodes)
# Create node features: [status, Worker_Counts, node_type]
node_features = [
    [G.nodes[node]["status"], G.nodes[node]["Worker_Counts"], G.nodes[node]["node_type"]]
    for node in node_names
]
x = torch.tensor(node_features, dtype=torch.float)

# Build edge_index from the NetworkX graph (mapping node names to indices)
edges = list(G.edges)
edge_index = torch.tensor(
    [[node_names.index(src), node_names.index(dst)] for src, dst in edges],
    dtype=torch.long
).t().contiguous()

data = Data(x=x, edge_index=edge_index)
print("Graph data:", data)

# Define the GAT Encoder with multi-head attention
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GATEncoder, self).__init__()
        # First GAT layer with multi-head attention
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second GAT layer: aggregate heads into output dimension
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

in_channels = data.num_features 
hidden_channels = 32
out_channels = 16

encoder = GATEncoder(in_channels, hidden_channels, out_channels, heads=8, dropout=0.6)
model = GAE(encoder)

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
model.train()
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "gat_model.pth")

# Evaluate the model and compute per-node reconstruction errors
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index)
    total_loss = model.recon_loss(z, data.edge_index)
    print("Total Reconstruction Loss:", total_loss.item())
    
    reconstructed_adj = torch.sigmoid(torch.matmul(z, z.t()))
    num_nodes = data.num_nodes
    original_adj = torch.zeros((num_nodes, num_nodes))
    original_adj[data.edge_index[0], data.edge_index[1]] = 1.0
    
    reconstruction_errors = ((original_adj - reconstructed_adj) ** 2).mean(dim=1)

machine1_errors = []
machine2_errors = []
machine3_errors = []
timestamps = list(range(num_rows))

for i in range(num_rows):
    node_m1 = f"M1_{i}"
    node_m2 = f"M2_{i}"
    node_m3 = f"M3_{i}"
    idx_m1 = node_names.index(node_m1)
    idx_m2 = node_names.index(node_m2)
    idx_m3 = node_names.index(node_m3)
    machine1_errors.append(reconstruction_errors[idx_m1].item())
    machine2_errors.append(reconstruction_errors[idx_m2].item())
    machine3_errors.append(reconstruction_errors[idx_m3].item())

plt.figure(figsize=(10, 6))
plt.plot(timestamps, machine1_errors, label="Machine 1", marker='o')
plt.plot(timestamps, machine2_errors, label="Machine 2", marker='o')
plt.plot(timestamps, machine3_errors, label="Machine 3", marker='o')
plt.xlabel("Time Step")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error per Machine Over Time")
plt.legend()
plt.grid(True)
plt.show()
