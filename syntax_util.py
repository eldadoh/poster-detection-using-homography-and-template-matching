import numpy as np
import os
import glob
import cv2 
import itertools
import functools

def iterating_using_zip_and_unpacking():
    
    a = np.random.randint(1,10,size =(3,3))
    b = np.where(a>3)
    for (y,x) in zip(*b):
        print(y,x)
    """    
    >>> a
        array([[5, 6, 2],               
              [5, 3, 5],
              [7, 3, 7]])
    
    >>> b
         (array([0, 0, 1, 1, 2, 2]), array([0, 1, 0, 2, 0, 2]))
    """
    
    """
    0 0
    0 1
    1 0
    1 2
    2 0
    2 2
    """