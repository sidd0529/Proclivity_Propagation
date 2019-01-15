from __future__ import division #For decimal division.
import numpy as np #For use in numerical computation.
from matplotlib import pylab as plt #Plotting.
import argparse #For commandline input
import scipy.io #For loading sparse matrices.
import sys
import time #Check time of computation.
import social_network_learn as snlearn #Custom designed function for train_test_split of nodes.
import mixing as mx #Custom designed function for computation of mixing matrices.
import feature_scaling as fs #Custom designed function for scaling of mixing matrices.
import prone as pr #Custom designed function for computation of prone scores.
from unique_elements import custom_unique #Custom designed function for finding appropriate unique array elements.
#np.set_printoptions(threshold=np.nan) #Uncomment if you want to print full numpy arrays.
from plots import visualize #Custom designed function for visualization of social network array.


# -------------------------------- Header/ metadata --------------------------------------------------------
__author__ = "Siddharth Satpathy"
__date__ = "15 January 2019"
__version__ = "1.0.0"
__email__ = "siddharthsatpathy.ss@gmail.com"
__status__ = "Complete"


# -------------------------------- Input arguments ---------------------------------------------------------
print "Usage: python proclivity.py -file Filename -prtype ProNe_Type -att Attribute_Num"
parser = argparse.ArgumentParser(description='Computes ProNe Correlation Matrix for Friendship Networks.')
parser.add_argument('-file', default='American75', help='Enter name of friendship network.')
parser.add_argument('-prtype', default='ProNel', help='Enter type of ProNe. Should be ProNel / ProNe2.')
parser.add_argument('-att', default='5', help='Enter attribute number.')


script_name = sys.argv[0][:-3]
args = parser.parse_args()
#print(args)

attribute_chosen = int(args.att)


# -------------------------------- Input -------------------------------------------------------------------
"""
Notes
----------
Input parameters.

Important parameters
----------
attributes : dictionary data type (dictionary of attributes)
nodes_original : numpy.ndarray data type (nodes-attributes array.)
edges : scipy sparse matrix (adjacency matrix with 0s and 1s.)	
edges_array : numpy.ndarray data type (adjacency matrix with 0s and 1s.)
keep_missing_values : True/False
					True -> Include zeroes in node while computing mixing matrices.
					False -> Ignore zeroes in node while computing mixing matrices.
"""
attributes = {0: 'ID', 1: 'Status', 2: 'Gender', 3: 'Major', 4: 'Minor', 5: 'Dorm', 6: 'Year', 7: 'High School'}

fname = scipy.io.loadmat('facebook100/' + args.file + '.mat')

nodes_original = fname['local_info']
edges = fname['A']
edges_array = edges.toarray()

keep_missing_values = False


# -------------------------------- Partition data into train and test datasets -----------------------------
nodes_train, nodes_test, label_train, label_test, test_indices = \
        snlearn.train_test_split(chosen_attribute=attribute_chosen, test_ratio=0.03, nodes=nodes_original)


# -------------------------------- Visualization of social network -----------------------------------------
visualize(nodes=nodes_test, node_indices=test_indices, 
	      chosen_attribute=attribute_chosen, adjacency=edges_array)


# -------------------------------- Mixing matrices ---------------------------------------------------------
"""
Notes
----------
Computation of mixing matrices

Important parameters
----------	
mix_mat : list data type. (stores mixing matrices between different pairs of attributes.)
mix_names : list data type. (assign names to mixing matrices between different pairs of attributes.) 
mix_dict : dictionary data type. (matches names from 'mix_names' with arrays from 'mix_mat'.)
"""
mix_mat = [ mx.mixing_matrix(attribute1=i, attribute2=j, nodes=nodes_train, 
	                         adjacency=edges_array, keep_missing_values=keep_missing_values) 
           for i in range(1, len(attributes.keys())) 
           for j in range(i, len(attributes.keys()))]

mix_names = [ 'mix'+str(i)+str(j) 
             for i in range(1, len(attributes.keys())) 
             for j in range(i, len(attributes.keys())) ]

mix_dict = { mix_names[i] : mix_mat[i] for i in range(len(mix_mat)) }

mix_dict.update( { 'mix'+str(j)+str(i) : np.transpose( mix_dict['mix'+str(i)+str(j)] )
             for i in range(1, len(attributes.keys())) 
             for j in range(i+1, len(attributes.keys())) }	)


# -------------------------------- Frequency of Attribute Values -------------------------------------------
"""
Important parameters
----------	
freq_attribute : list data type (stores frequencies of unique values for different attributes.)
freq_names : list data type (assign names to frequency arrays of different attributes.)
freq_dict : dictionary data type. (matches names from 'freq_names' with arrays from 'freq_attribute'.)
"""
freq_attribute = [ fs.frequency_scaler(nodes=nodes_train, chosen_attribute=i, 
	                                    keep_missing_values=keep_missing_values) 
                    for i in range(1, len(attributes.keys()))]

freq_names = [ 'freq_attribute'+str(i) for i in range(1, len(attributes.keys())) ] 

freq_dict = { freq_names[i] : freq_attribute[i] for i in range(len(freq_attribute)) }


# -------------------------------- Scaling of mixing matrices ----------------------------------------------
"""
Important parameters
----------	
scale_mix_mat : list data type. (stores scaled mixing matrices between different pairs of attributes.)
scale_mix_names : list data type. 
                (assign names to scaled mixing matrices between different pairs of attributes.) 
scale_mix_dict : dictionary data type. 
                (matches names from 'scale_mix_names' with arrays from 'scale_mix_mat'.)
"""
scale_mix_mat = [ mix_dict['mix'+str(i)+str(j)] / freq_dict['freq_attribute'+str(j)]  
           for i in range(1, len(attributes.keys())) 
           for j in range(1, len(attributes.keys()))]


scale_mix_names = [ 'scale_mix'+str(i)+str(j) 
            for i in range(1, len(attributes.keys())) 
            for j in range(1, len(attributes.keys())) ]

scale_mix_dict = { scale_mix_names[i] : scale_mix_mat[i] for i in range(len(scale_mix_mat)) }


# -------------------------------- ProNe matrix ------------------------------------------------------------
"""
Important parameters
----------	
ProNe_Mat : numpy.ndarray data type (stores prone scores.)
"""
ProNe_Mat = np.empty( (len(attributes.keys())-1, len(attributes.keys())-1) )
for i in range( 1, len(attributes.keys()) ):
    for j in range( i, len(attributes.keys()) ):
    	ProNe_Mat[i-1, j-1] = pr.choice( Mix=mix_dict['mix'+str(i)+str(j)], Type=args.prtype )
    	ProNe_Mat[j-1, i-1] = ProNe_Mat[i-1, j-1]


# -------------------------------- Proclivity propagation - prediction / score -----------------------------
"""
Important parameters
----------  
score : prediction accuracy.
confidence_score : confidence in prediction accuracy.
error_score : error in prediction accuracy.
"""
score, confidence_score = snlearn.proclivity_score( node_indices=test_indices, chosen_attribute=attribute_chosen, nodes=nodes_train, \
	   nodes_no_mask=nodes_original, adjacency=edges_array, prone_matrix=ProNe_Mat, attributes_dict=attributes,\
	   dict_scale_mix=scale_mix_dict, keep_missing_values=keep_missing_values)

error_score = 1-confidence_score


print "Accuracy for attribute ", attributes[attribute_chosen], " is: ", score, "."
print "Confidence in predicted accuracy is: ", confidence_score
print "Error in predicted accuracy is: ", error_score
print "Confidence given by uniform probability distribution is: ", \
    1/len( custom_unique(arr=nodes_train[:,attribute_chosen], keep_missing_values=keep_missing_values) ), "."


# snlearn.proclivity_predict( node_predict=test_indices[5], chosen_attribute=attribute_chosen, nodes=nodes_train, \
# 	                nodes_no_mask=nodes_original, adjacency=edges_array, prone_matrix=ProNe_Mat, attributes_dict=attributes, \
# 	                dict_scale_mix=scale_mix_dict, keep_missing_values=keep_missing_values )

