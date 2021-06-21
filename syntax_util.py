import numpy as np
import os
import glob
import cv2 
import itertools
import functools

def sort_list_by_elem():
    
    lis = [[9,4,7,'e'],[3,6,9,'d'],[2,59,8,'a']]
    lis_ = sorted(lis, key=lambda x: x[0]) 

    """
    [[2, 59, 8, 'a'], [3, 6, 9, 'd'], [9, 4, 7, 'e']]
    """

def display_zip_object(zip_item) : 

    length_ = len(list(zip_item))
    
    return length_,list(zip_item)

def iterating_with_zip_and_unpacking():
    
    a = np.random.randint(1,10,size =(3,3))
    
    b = np.where(a>3) 

    
    for (y,x) in zip(*b):
        print(y,x)
    """    
        a
        array([[5, 6, 2],               
              [5, 3, 5],
              [7, 3, 7]])
    
        b
         (array([0, 0, 1, 1, 2, 2]), array([0, 1, 0, 2, 0, 2]))
        
        output 
        
        0 0
        0 1
        1 0
        1 2
        2 0
        2 2

        print(type(b)) #tuple(nd.array , nd.array)

    """
