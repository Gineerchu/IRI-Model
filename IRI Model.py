import pandas as pd
import torch
import os
print("PyTorch has version {}".format(torch.__version__))
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.loader import NeighborSampler
import torch.nn.functional as F

# Load the GraphML file
lca_df = nx.read_graphml("G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/graph.graphml")
print(lca_df)

# Node (Intersection) Features
print(lca_df.number_of_nodes())
all_node_keys = list(list(lca_df.nodes[n].keys()) for n in lca_df.nodes())
all_node_keys = set(np.concatenate(all_node_keys).flat)
all_node_keys

# Edge (Road Segment) Features
print(lca_df.number_of_edges())
all_edge_keys = list(list(lca_df.edges[e].keys()) for e in lca_df.edges())
all_edge_keys = set(np.concatenate(all_edge_keys).flat)
all_edge_keys

# Average Degree
lca_df.number_of_edges() / lca_df.number_of_nodes()

# Histogram of out degree.
out_degrees = [v for (k,v) in lca_df.out_degree]
plt.hist(out_degrees, color = 'skyblue', lw=1, ec="black")

def featureVectorAndLabel(edge_attr):
    chosen_keys = ['feature_1', 'feature_2', 'feature_3']
    new_dict = {key: edge_attr[key] for key in chosen_keys if key in edge_attr}
    combined_list = list(new_dict.values())
    label = edge_attr['label']
    return combined_list, label

def getDualGraph(originalGraph):
  G = nx.DiGraph()
  for node in lca_df.nodes:
    incoming_edges = lca_df.in_edges(node, data=True)
    outgoing_edges = lca_df.out_edges(node, data=True)

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

G = getDualGraph(lca_df)

lca_df.number_of_edges()
G.number_of_nodes()
G.number_of_edges()

print(lca_df.number_of_edges() == G.number_of_nodes())


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

# Split into Train and Validation (80 / 20)
node_idx = label_mask.nonzero()
num_train = int(0.8 * len(node_idx))
shuffle = torch.randperm(node_idx.size(0))
train_idx = node_idx[shuffle[:num_train]]
val_idx = node_idx[shuffle[num_train:]]

print("Num train: ", len(train_idx))
print("Num val: ", len(val_idx))

train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[-1, -1], batch_size=32,
                               shuffle=True, num_workers=8, drop_last=True)

val_loader = NeighborSampler(data.edge_index, node_idx=val_idx,
                             sizes=[-1, -1], batch_size=32,
                             shuffle=True, num_workers=8, drop_last=True)

# Histogram of number of nodes per batch.
num_nodes = []
for _, n_id, _ in train_loader:
  num_nodes.append(len(n_id))
plt.hist(num_nodes, 50)
plt.show()

#-----------------------------------------------------Build GCN Model
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv

# layer_name should be one of:
#   SAGEConv
#   GATConv
class Model(torch.nn.Module):
    def __init__(self, layer_name, input_dim, hidden_dim, output_dim, edge_dim, num_layers, dropout):
        super().__init__()


        # Convolutional Layers
        self.layer_name = layer_name
        layers = []
        if layer_name == "SAGEConv":
          layers = [SAGEConv(input_dim, hidden_dim)]
          for _ in range(num_layers-2):
            layers.append(SAGEConv(hidden_dim, hidden_dim))
          layers.append(SAGEConv(hidden_dim, output_dim))

        elif layer_name == "GATConv":
          layers = [GATConv(input_dim, hidden_dim, edge_dim=edge_dim)]
          for _ in range(num_layers-2):
            layers.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
          layers.append(GATConv(hidden_dim, output_dim, edge_dim=edge_dim))
        
        self.convs = torch.nn.ModuleList(layers)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs, edge_attr):
      for i, (edge_index, e_id, size) in enumerate(adjs):
        x_target = x[:size[1]]  # Target nodes are always placed first.
        if self.layer_name == "SAGEConv":
          x = self.convs[i]((x, x_target), edge_index)
        elif self.layer_name == "GATConv":
          x = self.convs[i]((x, x_target), edge_index, edge_attr[e_id])
        if i < len(self.convs) - 1:
          x = F.relu(x)
          x = F.dropout(x, self.dropout, training=self.training)
      return x


# Returns dictionary of metrics with the following keys:
# correct: total items predicted correctly
# TP_k : # of True Positives for Label K
# FP_k : # of False Positives for Label K
# FN_k : # of False Negatives for Label K
def computeMetrics(predicted, actual):
  metrics = {}
  metrics['correct'] = int(predicted.eq(actual).sum())
  num_classes = 8
  for k in range(num_classes):
    positive_label = actual.eq(k)
    negative_label = torch.logical_not(positive_label)
    positive_prediction = predicted.eq(k) 
    negative_prediction = torch.logical_not(positive_prediction)

    TP = float(torch.logical_and(positive_label, positive_prediction).sum())
    FP = float(torch.logical_and(negative_label, positive_prediction).sum())
    FN = float(torch.logical_and(positive_label, negative_prediction).sum())

    metrics['TP_' + str(k)] = TP
    metrics['FP_' + str(k)] = FP
    metrics['FN_' + str(k)] = FN
    
  return metrics


# The Train function loops through all batches and computes the loss on the 
# target node.
def train(model, data, train_idx, train_loader, optimizer):
  model.train()
  total_loss = 0
  metrics = defaultdict(float)

  for batch_size, n_id, adjs in train_loader:
    # len(adjs) = number of hops
    adjs = [adj.to(device) for adj in adjs]
    labels = data.y[n_id[:batch_size]]
    labels = labels.to(torch.int64)
    labels_df=pd.DataFrame(labels)
    optimizer.zero_grad()
    out = model(data.x[n_id].float(), adjs, data.edge_attr.float())
    loss = F.cross_entropy(out, labels)
    loss.backward()
    optimizer.step()
    total_loss += float(loss)
    predictions = out.argmax(dim=-1)
    batch_metrics = computeMetrics(predictions, labels)
    # Bookkeeping
    metrics['total_correct'] += batch_metrics['correct']
    for k in range(8):
      metrics['total_TP_' + str(k)] += batch_metrics['TP_' + str(k)]
      metrics['total_FP_' + str(k)] += batch_metrics['FP_' + str(k)]
      metrics['total_FN_' + str(k)] += batch_metrics['FN_' + str(k)]

  metrics['total_loss'] = total_loss / len(train_loader)
  metrics['accuracy'] = metrics['total_correct'] / train_idx.size(0)
  for k in range(8):
    TP, FP, FN = metrics['total_TP_' + str(k)], metrics['total_FP_' + str(k)], metrics['total_FN_' + str(k)]
    metrics['precision_' + str(k)] = TP/(TP+FP) if TP+FP else 0
    metrics['recall_' + str(k)] = TP/(TP+FN) if TP+FN else 0

  return metrics


# The Validation function simply runs inference on each node in the validation
# set.
def validate(model, data, val_idx, val_loader):
  total_loss = 0
  metrics = defaultdict(float)

  for batch_size, n_id, adjs in val_loader:
    adjs = [adj.to(device) for adj in adjs]
    out = model(data.x[n_id].float(), adjs, data.edge_attr.float())
    labels = data.y[n_id[:batch_size]]
    labels = labels.to(torch.int64)
    labels_df=pd.DataFrame(labels)
    predictions = out.argmax(dim=-1)
    loss = F.cross_entropy(out, labels)
    total_loss += float(loss)
    batch_metrics = computeMetrics(predictions, labels)
    # Bookkeeping
    metrics['total_correct'] += batch_metrics['correct']
    for k in range(8):
      metrics['total_TP_' + str(k)] += batch_metrics['TP_' + str(k)]
      metrics['total_FP_' + str(k)] += batch_metrics['FP_' + str(k)]
      metrics['total_FN_' + str(k)] += batch_metrics['FN_' + str(k)]

  metrics['total_loss'] = total_loss / len(val_loader)
  metrics['accuracy'] = metrics['total_correct'] / val_idx.size(0)
  for k in range(8):
    TP, FP, FN = metrics['total_TP_' + str(k)], metrics['total_FP_' + str(k)], metrics['total_FN_' + str(k)]
    metrics['precision_' + str(k)] = TP/(TP+FP) if TP+FP else 0
    metrics['recall_' + str(k)] = TP/(TP+FN) if TP+FN else 0

  return metrics


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
hidden_dim = 32
num_layers = 2
dropout = 0.0
lr = 0.01
num_epochs = 100

# Dataset Parameters
num_features = 3
num_classes = 9
edge_dim = 2

# Model Instantiation
model = Model("SAGEConv", num_features, hidden_dim, num_classes, edge_dim, num_layers, dropout)
model.to(device)
model.reset_parameters()

# Use Adam optimizer to adapt learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


train_metrics_history = defaultdict(list)
val_metrics_history = defaultdict(list)
for epoch in range(1, 1 + num_epochs):
  print("Epoch: ", epoch)
  train_metrics = train(model, data, train_idx, train_loader, optimizer)
  val_metrics = validate(model, data, val_idx, val_loader)
  print("Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}".format(train_metrics['total_loss'], train_metrics['accuracy'], val_metrics['total_loss'], val_metrics['accuracy']))
  for metric_name, metric_value in train_metrics.items():
    train_metrics_history[metric_name].append(metric_value)
  for metric_name, metric_value in val_metrics.items():
    val_metrics_history[metric_name].append(metric_value)
    
    
# Accuracy, loss, precision, recall

#Train and Validation Loss
epochs = range(1, 1+num_epochs)
plt.plot(epochs, train_metrics_history['total_loss'], label = "Train Loss")
plt.plot(epochs, val_metrics_history['total_loss'], label = "Validation Loss")
plt.legend()
plt.show()
    
# Train and Validation Accuracy
epochs = range(1, num_epochs+1)
plt.plot(epochs, train_metrics_history['accuracy'], label = "Train Accuracy")
plt.plot(epochs, val_metrics_history['accuracy'], label = "Validation Accuracy")
plt.legend()
plt.show()

# Per-Class Precision and Recall
# Precision

epochs = range(1, num_epochs+1)
plt.figure(1, figsize=(10, 10))
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
for k in range(7):
  #plt.plot(epochs, train_metrics_history['precision_' + str(k)], label = "Train Precision # lanes = {}".format(k+1))
  plt.plot(epochs, val_metrics_history['precision_' + str(k)], label = "Validation Precision # lanes = {}".format(k+1))
plt.legend()
plt.show()

# With Smoothing
import numpy as np
import scipy.interpolate

epochs = range(1, num_epochs+1)


plt.figure(1, figsize=(10, 10))
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.ylim((0,1.0))
for k in range(7):
  x_new = np.linspace(1, 99, 5)
  spline = scipy.interpolate.make_interp_spline(epochs, val_metrics_history['precision_' + str(k)])
  y_new = spline(x_new)
  plt.plot(x_new, y_new, label = "Validation Precision # lanes = {}".format(k+1))
plt.legend()
plt.show()

# Recall
epochs = range(1, num_epochs+1)
plt.figure(1, figsize=(10, 10))
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
for k in range(7):
  #plt.plot(epochs, train_metrics_history['recall_' + str(k)], label = "Train Recall # lanes = {}".format(k+1))
  plt.plot(epochs, val_metrics_history['recall_' + str(k)], label = "Validation Recall # lanes = {}".format(k+1))
plt.legend()
plt.show()


# With Smoothing
import numpy as np
import scipy.interpolate

epochs = range(1, num_epochs+1)


plt.figure(1, figsize=(10, 10))
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.ylim((0,1.0))
for k in range(7):
  x_new = np.linspace(1, 99, 5)
  spline = scipy.interpolate.make_interp_spline(epochs, val_metrics_history['recall_' + str(k)])
  y_new = spline(x_new)
  plt.plot(x_new, y_new, label = "Validation Recall # lanes = {}".format(k+1))
plt.legend()
plt.show()
