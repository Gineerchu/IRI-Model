import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

 #import data
# merge_df = pd.read_csv("G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/fulldata/train_adj_AYA_woNA.csv")
merge_df = pd.read_csv("G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/fulldata/test_adj_AYA_woNA.csv")

# Sampling 100 rows
# sampled_data = merge_df.sample(n=100000,random_state=123)
sampled_data = merge_df
sampled_data['start_node'] = sampled_data['start_node'].astype(str)
sampled_data['stop_node'] = sampled_data['stop_node'].astype(str)
# sampled_data['lane_side'] = sampled_data['lane_side'].astype(str)

# Concatenating strings from two columns into a new column
# sampled_data['start_node'] = sampled_data['start_node'] + '-' + sampled_data['lane_side']
# sampled_data['stop_node'] = sampled_data['stop_node'] + '-' + sampled_data['lane_side']

sampled_data['start_node'] = sampled_data['start_node'] 
sampled_data['stop_node'] = sampled_data['stop_node'] 


# Mapping unique objects in 'node' to numbers
sampled_data['encoded_start_node'], _ = pd.factorize(sampled_data['start_node'])
sampled_data['encoded_stop_node'], _ = pd.factorize(sampled_data['stop_node'])

col = sampled_data.columns
multiple_columns = sampled_data[['start_node', 'stop_node', 'iri_2019','iri_2020','iri_2021','Y']].reset_index(drop=True)
multiple_columns = multiple_columns.fillna(0)
# Replace Values in Column (-1 = missing value)
multiple_columns['Y'] = multiple_columns['Y'].replace(0,-1)

multiple_columns = sampled_data 

# Filter rows with no NaN values in column 'Y'
filtered_df = multiple_columns[multiple_columns['Y'] >= 0]
# unique_values = filtered_df['section_id'].unique()
# Filter rows with NaN values in column 'Y'
nan_rows =  multiple_columns[multiple_columns['Y'] < 0]
# filtered_nan_rows = nan_rows[nan_rows['section_id'].isin(unique_values)]

# Concat not-na and na
# multiple_columns = pd.concat([filtered_df, filtered_nan_rows]).reset_index(drop=True)
# Drop section ID
# multiple_columns = multiple_columns.drop(columns=['section_id'])

df = multiple_columns
df.columns
df['Y'] = df['Y'].astype(float)
df = df.drop_duplicates()
df = df[df['Y'] > -1]

# df = df[df['Y'] > -1]

# count unique
start = df['start_node']
stop = df['stop_node']
com = pd.concat([start,stop])
from collections import Counter
count_elements = Counter(com)
num_unique_elements = len(count_elements)
print("Number of unique elements:", num_unique_elements)


# Creating an empty graph
G = nx.DiGraph()

# Adding edges with features
for index, row in df.iterrows():
    G.add_edge(row['start_node'], row['stop_node'], 
               feature_1=row['iri_2019'],
               feature_2=row['iri_2020'],
               feature_3=row['iri_2021'],
               label=row['Y'])
    

# Save the graph as a GraphML file
nx.write_graphml(G, "G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/train_adj_AYA_woNA.graphml")
# nx.write_graphml(G, "G:/Shared drives/2023 Projects/2023 DOH LCA/Paper/GCN/example data/test_adj_AYA_woNA.graphml")


G.number_of_nodes()
G.number_of_edges()

# # Create a plot of the graph 'G'
# nx.draw(G, with_labels=True)
# plt.show()

# Node (Intersection) Features
print(G.number_of_nodes())
all_node_keys = list(list(G.nodes[n].keys()) for n in G.nodes())
all_node_keys = set(np.concatenate(all_node_keys).flat)
all_node_keys

# Edge (Road Segment) Features
print(G.number_of_edges())
all_edge_keys = list(list(G.edges[e].keys()) for e in G.edges())
all_edge_keys = set(np.concatenate(all_edge_keys).flat)
all_edge_keys

