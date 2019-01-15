from __future__ import division
import numpy as np
from unique_elements import custom_unique #Custom designed function for finding appropriate unique array elements.


# -------------------------------- Feature scaling with frequency of unique values in nodes ----------------
def frequency_scaler(nodes, chosen_attribute, keep_missing_values):
    """
    Input parameters
    ----------
    nodes : numpy.ndarray data type (nodes-attributes array.)
    chosen_attribute : integer (represents a chosen attribute.)    
    keep_missing_values : True/False
                        True -> Include zeroes in node while computing mixing matrices.
                        False -> Ignore zeroes in node while computing mixing matrices.


    Important parameters used in function
    ----------
    nodes_cp : numpy.ndarray data type (copy of nodes.)
    num_nodes : integer (number of nodes.)
    attribute_unique : numpy.ndarray data type (unique values of chosen_attribute)
    frequency : numpy.ndarray data type (frequency of occurences of unique values of chosen_attribute)
    location_invalid_log : numpy.ndarray data type 
                     (indices where 'frequency=1'.
                     frequency=1 leads to problems when one takes log(frequency) ) 


    Returns
    ------- 
    np.log(frequency) / num_nodes : numpy.ndarray data type (frequencies of unique values of attributes.)
    """
    nodes_cp = nodes.copy()

    num_nodes = len( nodes_cp[:,0] )

    attribute_unique = custom_unique(arr=nodes_cp[:,chosen_attribute], keep_missing_values=keep_missing_values)

    frequency = np.zeros( len(attribute_unique) )

    for i in range( len(attribute_unique) ):
        mark = np.where( nodes_cp[:,chosen_attribute]==attribute_unique[i] )
        frequency[i] = len(mark[0])

    location_invalid_log = np.where(frequency==1)[0]
    frequency[ location_invalid_log ] = 2

    return np.log(frequency) / num_nodes