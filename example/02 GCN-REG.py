import pandas as pd
import torch
print("PyTorch has version {}".format(torch.__version__))
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.loader import NeighborSampler, DataLoader
import torch.nn.functional as F

# Load the GraphML file
newyork = nx.read_graphml("G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/train_adj_AYA_woNA.graphml")
print(newyork)

# Node (Intersection) Features
print(newyork.number_of_nodes())
all_node_keys = list(list(newyork.nodes[n].keys()) for n in newyork.nodes())
all_node_keys = set(np.concatenate(all_node_keys).flat)
all_node_keys

# Edge (Road Segment) Features
print(newyork.number_of_edges())
all_edge_keys = list(list(newyork.edges[e].keys()) for e in newyork.edges())
all_edge_keys = set(np.concatenate(all_edge_keys).flat)
all_edge_keys

# Average Degree
newyork.number_of_edges() / newyork.number_of_nodes()

# Histogram of out degree.
out_degrees = [v for (k,v) in newyork.out_degree]
plt.hist(out_degrees, color = 'skyblue', lw=1, ec="black")

def featureVectorAndLabel(edge_attr):
    chosen_keys = ['feature_1', 'feature_2', 'feature_3']
    new_dict = {key: edge_attr[key] for key in chosen_keys if key in edge_attr}
    combined_list = list(new_dict.values())
    label = edge_attr['label']
    return combined_list, label

def getDualGraph(originalGraph):
  G = nx.DiGraph()
  for node in newyork.nodes:
    incoming_edges = newyork.in_edges(node, data=True)
    outgoing_edges = newyork.out_edges(node, data=True)

    for (u_incoming, v_incoming, incoming_edge_attr) in incoming_edges:
      for (u_outgoing, v_outgoing, outgoing_edge_attr) in outgoing_edges:

        new_incoming_node = u_incoming + "-" + v_incoming
        new_outgoing_node = u_outgoing + "-" + v_outgoing

        if new_incoming_node not in G:
          G.add_node(new_incoming_node)
          x, y = featureVectorAndLabel(incoming_edge_attr)
          # x and y are node attributes.
          nx.set_node_attributes(G, {new_incoming_node: {'x': x, 'y': y}})

        if new_outgoing_node not in G:
          G.add_node(new_outgoing_node)
          x, y = featureVectorAndLabel(outgoing_edge_attr)
          nx.set_node_attributes(G, {new_outgoing_node: {'x': x, 'y': y}})

        G.add_edge(new_incoming_node, new_outgoing_node)
        original_in_degree = originalGraph.in_degree(node)
        original_out_degree = originalGraph.out_degree(node)
        
        # edge_attr is the edge attributes (in_degree and out_degree).
        nx.set_edge_attributes(G, {(new_incoming_node, new_outgoing_node):
        {'edge_attr': [original_in_degree, original_out_degree]}})
  return G

G = getDualGraph(newyork)

newyork.number_of_edges()
G.number_of_nodes()
G.number_of_edges()

print(newyork.number_of_edges() == G.number_of_nodes())

import networkx as nx

# Assuming 'G' is your graph obtained from getDualGraph or another method

# Get the NodeView of the graph G
nodes_view = G.nodes

# Convert NodeView to a list and access the first node
node_list_train = list(nodes_view)
if node_list_train:  # Check if the list is not empty
    first_node = node_list_train[0]
    print(f"First Node in the graph: {first_node}")
    # If you want the attributes of the first node
    first_node_attributes = G.nodes[first_node]
    print("Attributes of the first node:")
    for attribute, value in first_node_attributes.items():
        print(f"{attribute}: {value}")
else:
    print("The graph does not contain any nodes.")

#-----------------------------------------------------------------------------
# Converts NetworkX graph to torch_geometric.data.Data
data = torch_geometric.utils.convert.from_networkx(G)
e_a = pd.DataFrame(data['edge_attr'])
e_index = pd.DataFrame(data['edge_index'])
y_node = pd.DataFrame(data['y'])
x_node = pd.DataFrame(data['x'])

print(type(data))
data.num_node_features

data.node_attrs()
data.num_edges

labels = torch.masked_select(data.y, torch.where(data.y >= 0, True, False))
labels_df=pd.DataFrame(labels)
# plt.hist(labels, bins=7, color = 'skyblue', lw=1, ec="black")
# Nodes with lane count label present.
label_mask = torch.where(data.y >= 0, True, False)

# 11,842 total nodes with a label.
torch.sum(label_mask == True)

node_idx = label_mask.nonzero()
num_train = int(1 * len(node_idx))
shuffle = torch.randperm(node_idx.size(0))
train_idx = node_idx[shuffle[:num_train]]

train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[-1, -1], batch_size=64,
                               shuffle=True, num_workers=8, drop_last=True)


# Load the GraphML file
newyork = nx.read_graphml("G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/test_adj_AYA_woNA.graphml")
print(newyork)

# Node (Intersection) Features
print(newyork.number_of_nodes())
all_node_keys = list(list(newyork.nodes[n].keys()) for n in newyork.nodes())
all_node_keys = set(np.concatenate(all_node_keys).flat)
all_node_keys

# Edge (Road Segment) Features
print(newyork.number_of_edges())
all_edge_keys = list(list(newyork.edges[e].keys()) for e in newyork.edges())
all_edge_keys = set(np.concatenate(all_edge_keys).flat)
all_edge_keys

# Average Degree
newyork.number_of_edges() / newyork.number_of_nodes()

# Histogram of out degree.
out_degrees = [v for (k,v) in newyork.out_degree]
plt.hist(out_degrees, color = 'skyblue', lw=1, ec="black")

def featureVectorAndLabel(edge_attr):
    chosen_keys = ['feature_1', 'feature_2', 'feature_3']
    new_dict = {key: edge_attr[key] for key in chosen_keys if key in edge_attr}
    combined_list = list(new_dict.values())
    label = edge_attr['label']
    return combined_list, label

def getDualGraph(originalGraph):
  G = nx.DiGraph()
  for node in newyork.nodes:
    incoming_edges = newyork.in_edges(node, data=True)
    outgoing_edges = newyork.out_edges(node, data=True)

    for (u_incoming, v_incoming, incoming_edge_attr) in incoming_edges:
      for (u_outgoing, v_outgoing, outgoing_edge_attr) in outgoing_edges:

        new_incoming_node = u_incoming + "-" + v_incoming
        new_outgoing_node = u_outgoing + "-" + v_outgoing

        if new_incoming_node not in G:
          G.add_node(new_incoming_node)
          x, y = featureVectorAndLabel(incoming_edge_attr)
          # x and y are node attributes.
          nx.set_node_attributes(G, {new_incoming_node: {'x': x, 'y': y}})

        if new_outgoing_node not in G:
          G.add_node(new_outgoing_node)
          x, y = featureVectorAndLabel(outgoing_edge_attr)
          nx.set_node_attributes(G, {new_outgoing_node: {'x': x, 'y': y}})

        G.add_edge(new_incoming_node, new_outgoing_node)
        original_in_degree = originalGraph.in_degree(node)
        original_out_degree = originalGraph.out_degree(node)
        
        # edge_attr is the edge attributes (in_degree and out_degree).
        nx.set_edge_attributes(G, {(new_incoming_node, new_outgoing_node):
        {'edge_attr': [original_in_degree, original_out_degree]}})
  return G

G = getDualGraph(newyork)

newyork.number_of_edges()
G.number_of_nodes()
G.number_of_edges()

print(newyork.number_of_edges() == G.number_of_nodes())

import networkx as nx

# Assuming 'G' is your graph obtained from getDualGraph or another method

# Get the NodeView of the graph G
nodes_view = G.nodes

# Convert NodeView to a list and access the first node
node_list_test = list(nodes_view)
if node_list_test:  # Check if the list is not empty
    first_node = node_list_test[0]
    print(f"First Node in the graph: {first_node}")
    # If you want the attributes of the first node
    first_node_attributes = G.nodes[first_node]
    print("Attributes of the first node:")
    for attribute, value in first_node_attributes.items():
        print(f"{attribute}: {value}")
else:
    print("The graph does not contain any nodes.")


#-----------------------------------------------------------------------------
# Converts NetworkX graph to torch_geometric.data.Data
data_test = torch_geometric.utils.convert.from_networkx(G)
e_a = pd.DataFrame(data_test['edge_attr'])
e_index = pd.DataFrame(data_test['edge_index'])
y_node = pd.DataFrame(data_test['y'])
x_node = pd.DataFrame(data_test['x'])

print(type(data))
data_test.num_node_features

data_test.node_attrs()
data_test.num_edges

labels = torch.masked_select(data_test.y, torch.where(data_test.y >= 0, True, False))
labels_df=pd.DataFrame(labels)
# plt.hist(labels, bins=7, color = 'skyblue', lw=1, ec="black")
# Nodes with lane count label present.
label_mask = torch.where(data_test.y >= 0, True, False)

# 11,842 total nodes with a label.
torch.sum(label_mask == True)


node_idx = label_mask.nonzero()
# node_idx=pd.DataFrame(node_idx)
num_train = int(1 * len(node_idx))
shuffle = torch.randperm(node_idx.size(0))
test_idx = node_idx[shuffle[:num_train]]

test_loader = NeighborSampler(data_test.edge_index, node_idx=test_idx,
                             sizes=[-1, -1], batch_size=32,
                             shuffle=True, num_workers=8, drop_last=True)

#-----------------------------------------------------Build GCN Model
from torch_geometric.nn import SAGEConv

# layer_name should be one of:
#   SAGEConv
class Model(torch.nn.Module):
    def __init__(self, layer_name, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()

        # Convolutional Layers
        self.layer_name = layer_name
        layers = []
        if layer_name == "SAGEConv":
          layers = [SAGEConv(input_dim, hidden_dim)]
          for _ in range(num_layers-2):
            layers.append(SAGEConv(hidden_dim, hidden_dim))
          layers.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)  # Adding a linear layer
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs, edge_attr):
      for i, (edge_index, e_id, size) in enumerate(adjs):
        x_target = x[:size[1]]  # Target nodes are always placed first.
        if self.layer_name == "SAGEConv":
          x = self.convs[i]((x, x_target), edge_index)
        if i < len(self.convs) - 1:
          x = F.relu(x)
          x = F.dropout(x, self.dropout, training=self.training)
     # Adding the linear layer with ReLU activation
      x = self.linear(x)
      x = F.relu(x)
      return x


# Returns dictionary of metrics with the following keys:

def computeRegressionMetrics(predicted, actual):
    metrics = {}
    predicted = predicted.float()
    actual = actual.float()
    # Mean Squared Error (MSE) calculation
    squared_error = (predicted - actual) ** 2
    mse = torch.mean(squared_error)
    metrics['MSE'] = mse.item()
    
    # Mean Absolute Error (MAE) calculation
    abs_error = torch.abs(predicted - actual)
    mae = torch.mean(abs_error)
    metrics['MAE'] = mae.item()

    # Mean Absolute Percentage Error (MAPE) calculation
    abs_percentage_error = torch.abs((actual - predicted) / actual)
    mape = torch.mean(abs_percentage_error) * 100
    metrics['MAPE'] = mape.item()

    # R-squared (Coefficient of Determination) calculation
    ss_res = torch.sum((actual - predicted) ** 2)
    ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R^2'] = r2.item()

    return metrics


# The Train function loops through all batches and computes the loss on the 
# target node.
def train(model, data, train_idx, train_loader, optimizer):
  model.train()
  total_loss = 0
  all_labels = []
  all_outputs = []
  metrics = defaultdict(float)
  # for node_id in node_list_train:
  #   print(node_id)
  for batch_size, n_id, adjs in train_loader:
        # len(adjs) = number of hops
        adjs = [adj.to(device) for adj in adjs]
        labels = data.y[n_id[:batch_size]]
        labels = labels.unsqueeze(1)
        labels = labels.to(float)
        labels_df=pd.DataFrame(labels)
        optimizer.zero_grad()
        
        out = model(data.x[n_id].float(), adjs, data.edge_attr.float())
        out = out.to(float)
        loss = F.mse_loss(out, labels)
        out_df = pd.DataFrame(out.detach().numpy())
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        
        # Accumulate labels and outputs
        all_labels.append(labels.detach().numpy())
        all_outputs.append(out.detach().numpy())
        
        
        batch_metrics = computeRegressionMetrics(out, labels)
    
        # Bookkeeping
        metrics['total_MSE'] += batch_metrics['MSE']
        metrics['total_MAE'] += batch_metrics['MAE']
        metrics['total_MAPE'] += batch_metrics['MAPE']
        metrics['total_R^2'] += batch_metrics['R^2']
    
        # Calculate averages or final metrics after processing batches (assuming train_loader batches)
    
  # Concatenate all labels and outputs
  all_labels = np.concatenate(all_labels)
  all_outputs = np.concatenate(all_outputs)

  labels_df = pd.DataFrame(all_labels, columns=['actual'])
  out_df = pd.DataFrame(all_outputs, columns=['predicted'])

  # Concatenate labels and outputs DataFrames
  combined_df = pd.concat([labels_df, out_df], axis=1)

  num_batches = len(train_loader)
    
  metrics['total_MSE'] /= num_batches
  metrics['total_MAE'] /= num_batches
  metrics['total_MAPE'] /= num_batches
  metrics['total_R^2'] /= num_batches

  return metrics, combined_df



# The test function simply runs inference on each node in the validation
# set.
def test(model, data, test_idx, test_loader):
  total_loss = 0
  all_labels = []
  all_outputs = []
  metrics = defaultdict(float)

  for batch_size, n_id, adjs in test_loader:
        # len(adjs) = number of hops
        adjs = [adj.to(device) for adj in adjs]
        labels = data.y[n_id[:batch_size]]
        labels = labels.unsqueeze(1)
        labels = labels.to(float)
        labels_df=pd.DataFrame(labels)
        
        out = model(data.x[n_id].float(), adjs, data.edge_attr.float())
        out = out.to(float)
        loss = F.mse_loss(out, labels)
        out_df = pd.DataFrame(out.detach().numpy())
        
        total_loss += float(loss)
        # Accumulate labels and outputs
        all_labels.append(labels.detach().numpy())
        all_outputs.append(out.detach().numpy())
        
        
        batch_metrics = computeRegressionMetrics(out, labels)
    
        # Bookkeeping
        metrics['total_MSE'] += batch_metrics['MSE']
        metrics['total_MAE'] += batch_metrics['MAE']
        metrics['total_MAPE'] += batch_metrics['MAPE']
        metrics['total_R^2'] += batch_metrics['R^2']


    # Calculate averages or final metrics after processing batches (assuming train_loader batches)
  # Concatenate all labels and outputs
  all_labels = np.concatenate(all_labels)
  all_outputs = np.concatenate(all_outputs)

  labels_df = pd.DataFrame(all_labels, columns=['actual'])
  out_df = pd.DataFrame(all_outputs, columns=['predicted'])

  # Concatenate labels and outputs DataFrames
  combined_df = pd.concat([labels_df, out_df], axis=1)
  num_batches = len(train_loader)
    
  metrics['total_MSE'] /= num_batches
  metrics['total_MAE'] /= num_batches
  metrics['total_MAPE'] /= num_batches
  metrics['total_R^2'] /= num_batches

  return metrics, combined_df

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
hidden_dim = 50
num_layers = 2
dropout = 0.0
lr = 0.001
num_epochs = 30

# Dataset Parameters
num_features = 3
output_dim = 1

# Model Instantiation
model = Model("SAGEConv", num_features, hidden_dim, output_dim, num_layers, dropout)
model.to(device)
model.reset_parameters()

# Use Adam optimizer to adapt learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


train_metrics_history = defaultdict(list)
val_metrics_history = defaultdict(list)
test_metrics_history = defaultdict(list)
for epoch in range(1, 1 + num_epochs):
  print("Epoch: ", epoch)
  train_metrics, train_df = train(model, data, train_idx, train_loader, optimizer)
  test_metrics, test_df = test(model, data_test, test_idx, test_loader)
  
  train_MSE, train_MAE, train_MAPE, train_R2 = train_metrics['total_MSE'], train_metrics['total_MAE'], train_metrics['total_MAPE'], train_metrics['total_R^2']
  test_MSE, test_MAE, test_MAPE, test_R2 = test_metrics['total_MSE'], test_metrics['total_MAE'], test_metrics['total_MAPE'], test_metrics['total_R^2']
  
  print(f"Train MSE: {train_MSE}, Train MAE: {train_MAE}, Train MAPE: {train_MAPE}, Train R^2: {train_R2}, Test MSE: {test_MSE}, Test MAE: {test_MAE}, Test MAPE: {test_MAPE}, Test R^2: {test_R2}")

  
  for metric_name, metric_value in train_metrics.items():
    train_metrics_history[metric_name].append(metric_value)
    
  for metric_name, metric_value in test_metrics.items():
    test_metrics_history[metric_name].append(metric_value) 
    
    
#Train and Test Loss
epochs = range(1, 1+num_epochs)
plt.plot(epochs, train_metrics_history['total_MAPE'], label = "Training")
plt.plot(epochs, test_metrics_history['total_MAPE'], label = "Test")
plt.xlim(0, num_epochs + 1)
# plt.ylim(0, 40)
plt.legend()
plt.show()
    
train_df.to_csv('G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/result_train_AYA_woNA.csv',index=False,header=True)
test_df.to_csv('G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/result_test_AYA_woNA.csv',index=False,header=True)

