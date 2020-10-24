from __future__ import division #For decimal division.
import numpy as np #For use in numerical computation.
import networkx as nx


# -------------------------------- Visualization of social network -----------------------------------------
def visualize(nodes, node_indices, chosen_attribute, adjacency):
	"""
	Input parameters
	----------
	nodes : numpy.ndarray data type (training nodes-attributes array.)
	node_indices : numpy.ndarray data type (indices of nodes for which prediction is being tested.)	               
	chosen_attribute : integer (represents a chosen attribute.)	               
	adjacency : numpy.ndarray data type (adjacency matrix with 0s and 1s.)	


	Important parameters used in function
	----------
	nodes_visualize : numpy.ndarray data type (copy of nodes.)
	adjacency_cp : numpy.ndarray data type (copy of adjacency.)	
	"""	
	nodes_visualize = nodes.copy()
	edges_visualize = (adjacency[node_indices, :])[:,node_indices]


	attribute_values = nodes_visualize[:,chosen_attribute]

	attribute_to_color = dict()
	for ii, attribute in enumerate(np.unique(attribute_values)):
	    attribute_to_color[attribute] = np.random.rand(3)
	    
	node_color = [attribute_to_color[attribute] for attribute in attribute_values]  

	node_size = nodes_visualize[:, 2] * 10
	G = nx.from_numpy_matrix(edges_visualize)
	nx.draw(G, node_color=node_color, node_size=node_size, with_labels=True, font_size=0)
	nx.write_gml(G, "visualize_social graph.gml")
