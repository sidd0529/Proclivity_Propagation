from __future__ import division
import numpy as np


# -------------------------------- ProNel ------------------------------------------------------------------
def ProNel(Mix):          
    """
    Input parameters
    ----------
    Mix : numpy.ndarray data type (mixing matrix.)


    Returns
    ----------
    ProNel_ret : float (ProNel score for the mixing matrix.)
    """
    Mix = Mix/np.sum(Mix[:,:])

    Mix_MargR = Mix.sum(axis=0)
    Mix_MargR = Mix_MargR[ np.nonzero(Mix_MargR)[0] ]
    Mix_MargC = Mix.sum(axis=1)
    Mix_MargC = Mix_MargC[ np.nonzero(Mix_MargC)[0] ]

    Num_T1 = Mix_MargR * np.log(Mix_MargR)
    Num_T1 = np.sum(Num_T1)

    Num_T2 = Mix_MargC * np.log(Mix_MargC)
    Num_T2 = np.sum(Num_T2)

    MixNz = Mix[np.nonzero(Mix)[0],np.nonzero(Mix)[1]]
    Num_T3 = np.multiply( MixNz, np.log(MixNz) )
    Num_T3 = np.sum(Num_T3)

    Num = Num_T1 + Num_T2 - Num_T3

    Den_T1 = 0.5*Num_T1
    Den_T2 = 0.5*Num_T2
    Den = Den_T1 + Den_T2

    ProNel_ret = Num / Den
    return ProNel_ret


# -------------------------------- ProNe2 ------------------------------------------------------------------
def ProNe2(Mix): 
    """
    Input parameters
    ----------
    Mix : numpy.ndarray data type (mixing matrix.)


    Returns
    ----------
    ProNe2_ret : float (ProNe2 score for the mixing matrix.)
    """
    Mix = Mix/np.sum(Mix[:,:])  

    Mix_MargR = Mix.sum(axis=0)
    Mix_MargR = Mix_MargR[ np.nonzero(Mix_MargR)[0] ]
    Mix_MargC = Mix.sum(axis=1)
    Mix_MargC = Mix_MargC[ np.nonzero(Mix_MargC)[0] ]

    Num_T1 = np.multiply( Mix, Mix )
    Num_T1 = np.sum(Num_T1)

    Num_T2 = np.sum(Mix_MargC*Mix_MargC) * np.sum(Mix_MargR*Mix_MargR) 
    
    Num = Num_T1 - Num_T2

    Den_T1 = 0.5 * (np.sum(Mix_MargC*Mix_MargC) + np.sum(Mix_MargR*Mix_MargR))
    Den_T2 = Num_T2
    Den = Den_T1 - Den_T2

    ProNe2_ret = Num / Den
    return ProNe2_ret


# -------------------------------- Choice of ProNe ---------------------------------------------------------  
def choice(Mix, Type): 
    """
    Input parameters
    ----------
    Mix : numpy.ndarray data type (mixing matrix.)
    Type : string data type ('ProNel' or 'ProNe2')
    """    
    if Type == 'ProNel':
        return ProNel(Mix=Mix)
    if Type == 'ProNe2':
        return ProNe2(Mix=Mix)        