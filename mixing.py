from __future__ import division
import numpy as np
from unique_elements import custom_unique #Custom designed function for finding appropriate unique array elements.


# -------------------------------- Compute mixing matrices -------------------------------------------------
def mixing_matrix(attribute1, attribute2, nodes, adjacency, keep_missing_values):
    """
    Input parameters
    ----------
    attribute1 : integer (represents the first chosen attribute.)
    attribute2 : integer (represents the second chosen attribute.)
    nodes : numpy.ndarray data type (nodes-attributes array.)
    adjacency : numpy.ndarray data type (adjacency matrix.)
    keep_missing_values : True/False
                  True -> Include zeroes in node while computing mixing matrices.
                  False -> Ignore zeroes in node while computing mixing matrices.


    Important parameters used in function
    ----------
    nodes_cp : numpy.ndarray data type (copy of nodes.)
    attribute1_unique : numpy.ndarray data type (unique values of attribute1)
    attribute2_unique : numpy.ndarray data type (unique values of attribute2)
    attribute1_unique_indices : numpy.ndarray data type (indices of unique values of attribute1)
    attribute2_unique_indices : numpy.ndarray data type (indices of unique values of attribute2)


    Returns
    ----------
    mix : numpy.ndarray data type (mixing matrix.)
    """
    nodes_cp = nodes.copy()

    attribute1_unique = custom_unique(arr=nodes_cp[:,attribute1], keep_missing_values=keep_missing_values)
    attribute2_unique = custom_unique(arr=nodes_cp[:,attribute2], keep_missing_values=keep_missing_values)

    mix = np.zeros( [ len(attribute1_unique) , len(attribute2_unique) ] )

    for i in range( 0, len(attribute1_unique) ):
        attribute1_unique_indices = np.where( nodes_cp[:,attribute1]==attribute1_unique[i] )[0]
        for j in range( 0, len(attribute2_unique) ):
            attribute2_unique_indices = np.where( nodes_cp[:,attribute2]==attribute2_unique[j] )[0]
            mix[i,j] = \
             len( np.nonzero( adjacency[ np.ix_( attribute1_unique_indices , attribute2_unique_indices ) ] )[0] )/2 

    return mix