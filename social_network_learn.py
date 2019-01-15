from __future__ import division
import numpy as np
from unique_elements import custom_unique #Custom designed function for finding appropriate unique array elements.
#np.set_printoptions(threshold=np.nan) #Uncomment if you want to print full numpy arrays.


# -------------------------------- Partition data into train and test datasets -----------------------------
def train_test_split(chosen_attribute, test_ratio, nodes):
	"""
	Input parameters
	----------
	chosen_attribute : integer (represents a chosen attribute.)
	test_ratio	: float 
	            (Less than one. 
	            Represents fraction of appropriate nodes which should be used for creation of test data set.)
	nodes : numpy.ndarray data type (nodes-attributes array.)


	Important parameters used in function
	----------
	indices_nonzero_att_value : numpy.ndarray data type 
	                  (contains indices of nodes with non zero values of 'chosen_attribute'.)
	test_num : integer (number of nodes in test data set.)
	test_indices : numpy.ndarray data type (indices of nodes in test data set.)
	nodes_original_cp : numpy.ndarray data type (copy of nodes.)
	nodes_test : numpy.ndarray data type (nodes for test data set.)
	nodes_train : numpy.ndarray data type (nodes for training data set.)
	

	Returns
	------- 
	nodes_test : numpy.ndarray data type (nodes for test data set.)
	nodes_train : numpy.ndarray data type (nodes for training data set.)
	nodes[:, chosen_attribute] : numpy.ndarray data type (labels for training data set.)
	nodes[test_indices, chosen_attribute] : numpy.ndarray data type (labels for test data set.)
	"""
	indices_nonzero_att_value = np.where( nodes[:,chosen_attribute] != 0 )[0]
	test_num = int( len(indices_nonzero_att_value)*test_ratio )
	test_indices = np.random.choice( indices_nonzero_att_value, size=test_num, replace=False )

	nodes_original_cp = nodes.copy()

	nodes_test = nodes_original_cp[test_indices, :]    

	nodes_train = nodes_original_cp
	nodes_train[test_indices, chosen_attribute] = 0    

	return nodes_train, nodes_test, nodes[:, chosen_attribute], nodes[test_indices, chosen_attribute], test_indices


# -------------------------------- Proclivity propagation - prediction -------------------------------------
def proclivity_predict( node_predict, chosen_attribute, nodes, nodes_no_mask, 
	             adjacency, prone_matrix, attributes_dict, dict_scale_mix, keep_missing_values ):
	"""
	Input parameters
	----------
	node_predict : integer (index of node whose attribute value is to be predicted.)
	chosen_attribute : integer (represents a chosen attribute.)
	nodes : numpy.ndarray data type (training nodes-attributes array.)
	nodes_no_mask : numpy.ndarray data type 
	               (nodes-attributes array where test attribute values have not been masked, 
	               i.e. original nodes-attributes array.)
	adjacency : numpy.ndarray data type (adjacency matrix with 0s and 1s.)
	prone_matrix : numpy.ndarray data type  (prone matrix.)
	attributes_dict : dictionary data type (dictionary of attributes)
	dict_scale_mix : dictionary data typ. 
                (matches names from 'scale_mix_names' with arrays from 'scale_mix_mat'.)
	keep_missing_values : True/False
                       True -> Include zeroes in node while computing mixing matrices.
                       False -> Ignore zeroes in node while computing mixing matrices.


	Important parameters used in function
	----------
	nodes_cp : numpy.ndarray data type (copy of nodes.)
	nodes_no_mask_cp : numpy.ndarray data type (copy of nodes_no_mask.)
	neighbors : numpy.ndarray data type (neighbors of node_predict.)
	prone_predictor : list data type (attribute value chosen for prediction based on prone matrix.)
	attribute_chosen_unique : numpy.ndarray data type (unique values of chosen_attribute.)
	attribute_predicting_unique : numpy.ndarray data type (unique values of the (predicting) attribute.)
	predicted_attribute_value : float (predicted attribute value.)
	true_attribute_value : float (true attribute value.)
	confidence : float (confidence in prediction of true_attribute_value.)
	

	Returns
	------- 
	nodes_test : numpy.ndarray data type (nodes for test data set.)
	"""	
	nodes_cp = nodes.copy()
	nodes_no_mask_cp = nodes_no_mask.copy()
	adjacency_cp = adjacency.copy()
	dict_scale_mix_cp = dict_scale_mix.copy()

	neighbors = ( np.nonzero( adjacency_cp[ node_predict , : ] ) )[0]

	attribute_chosen_unique = custom_unique(arr=nodes_cp[:,chosen_attribute], keep_missing_values=keep_missing_values)

	Total_Weight = np.zeros( len(attribute_chosen_unique) )
	
	pred_dict = { 'pred_from_att'+str(i) : np.zeros( len(attribute_chosen_unique) ) 
	              for i in range( 1, len(attributes_dict.keys()) ) }

	""" Loop over attributes. """
	use_prone = True

	if (use_prone):
		prone_predictor = [np.argmax( prone_matrix[chosen_attribute-1,:] )+1] 

		if (chosen_attribute!=7) and (prone_predictor==7): 
			prone_predictor=range( chosen_attribute, chosen_attribute+1 )

		
	else: prone_predictor = range( chosen_attribute, chosen_attribute+1 )


	#for pred_att in range( 1, len(attributes_dict.keys()) ):	
	for pred_att in prone_predictor:
		attribute_predicting_unique = custom_unique(arr=nodes_cp[:,pred_att], keep_missing_values=keep_missing_values)

		""" Loop over neighbors. """
		for j in range( len(neighbors) ):
			neighbor_att_value = nodes_cp[ neighbors[j] , pred_att ]
			if (neighbor_att_value!=0):
				neighbor_att_value_location = np.where( attribute_predicting_unique == neighbor_att_value )[0][0]

				mix_row = \
				 (dict_scale_mix_cp['scale_mix'+str(pred_att)+str(chosen_attribute)])[ neighbor_att_value_location,: ]

				(pred_dict['pred_from_att'+str(pred_att)])[np.argmax(mix_row)] += 1

		magnify = 100
		Total_Weight = Total_Weight + ( magnify * pred_dict['pred_from_att'+str(pred_att)] )

		if np.sum(Total_Weight) != 0:
			predicted_attribute_value = attribute_chosen_unique[ np.argmax(Total_Weight) ]
			true_attribute_value = nodes_no_mask_cp[ node_predict, chosen_attribute ]

			if (true_attribute_value in attribute_chosen_unique): 
				location_true_value = np.where( attribute_chosen_unique==true_attribute_value )[0][0]
				confidence = Total_Weight[ location_true_value ] / np.sum(Total_Weight)

				return predicted_attribute_value, confidence

			else: return -1, -1	


		if np.sum(Total_Weight) == 0: return -1, -1	


# -------------------------------- Proclivity propagation - score ------------------------------------------
def proclivity_score(node_indices, chosen_attribute, nodes, nodes_no_mask, 
	            adjacency, prone_matrix, attributes_dict, dict_scale_mix, keep_missing_values):
	"""
	Input parameters
	----------
	node_indices : numpy.ndarray data type (indices of nodes for which prediction is being tested.)
	chosen_attribute : integer (represents a chosen attribute.)
	nodes_no_mask : numpy.ndarray data type 
	               (nodes-attributes array where test attribute values have not been masked, 
	               i.e. original nodes-attributes array.)
	adjacency : numpy.ndarray data type (adjacency matrix with 0s and 1s.)
	prone_matrix : numpy.ndarray data type  (prone matrix.)
	attributes_dict : dictionary data type (dictionary of attributes)
	dict_scale_mix : dictionary data typ. 
                (matches names from 'scale_mix_names' with arrays from 'scale_mix_mat'.)
	keep_missing_values : True/False
                       True -> Include zeroes in node while computing mixing matrices.
                       False -> Ignore zeroes in node while computing mixing matrices.


	Important parameters used in function
	----------
	nodes_cp : numpy.ndarray data type (copy of nodes.)
	nodes_no_mask_cp : numpy.ndarray data type (copy of nodes_no_mask.)
	adjacency_cp : numpy.ndarray data type (copy of adjacency.)
	dict_scale_mix_cp : dictionary data type (copy of dict_scale_mix.)
	score : prediction accuracy.
	confidence_score : confidence in prediction accuracy.


	Returns
	------- 
	nodes_test : numpy.ndarray data type (nodes for test data set.)
	"""
	nodes_cp = nodes.copy()
	nodes_no_mask_cp = nodes_no_mask.copy()
	adjacency_cp = adjacency.copy()
	dict_scale_mix_cp = dict_scale_mix.copy()

	true_attribute_values = nodes_no_mask_cp[ node_indices, chosen_attribute ]

	result_list = []
	confidence_list = []

	for i in range( len(node_indices) ):
		predicted_attribute_value, confidence = proclivity_predict( node_predict=node_indices[i], \
		    chosen_attribute=chosen_attribute, nodes=nodes_cp, nodes_no_mask=nodes_no_mask_cp, adjacency=adjacency_cp, \
		    prone_matrix=prone_matrix, attributes_dict=attributes_dict, dict_scale_mix=dict_scale_mix_cp, \
		    keep_missing_values=keep_missing_values )


		if (predicted_attribute_value != -1):
			if (predicted_attribute_value == true_attribute_values[i]):
				result_list += [1]
				confidence_list += [confidence]

			if (predicted_attribute_value != true_attribute_values[i]):	
				result_list += [0]
				confidence_list += [confidence]

	score = np.sum(result_list)/len(result_list)
	confidence_score = np.mean(confidence_list)

	return score, confidence_score