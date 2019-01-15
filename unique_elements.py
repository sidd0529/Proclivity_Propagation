import numpy as np


# -------------------------------- Returns appropriate unique elements of arrays ---------------------------
def custom_unique(arr, keep_missing_values):
	"""
	Input parameters
	----------
	arr : numpy.ndarray data type
	keep_missing_values : True/False
					True -> Include zeroes in node while computing mixing matrices.
					False -> Ignore zeroes in node while computing mixing matrices.


	Returns
	------- 
	unique_entries_array : numpy.ndarray data type (customized appropriate unique entries of 'arr'.)


	Examples
	--------
	>>> arr2 = np.array([2, 3, 6, 1, 2, 1, 2, 0, 2, 0])
	>>> custom_unique(arr=arr2, keep_missing_values=False)

	array([1, 2, 3, 6])
	"""		
	unique_entries_array = np.unique(arr)

	if keep_missing_values == False:
		unique_entries_array = unique_entries_array[ (np.nonzero(unique_entries_array)[0]) ]

	return unique_entries_array