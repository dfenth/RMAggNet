"""
Utility functions for analysing the generated sets and
calculating Hamming distance.
"""

def analyse_sets(class_bitstrings):
    """
    Tells us the Hamming distance between each class bitstring
    
    Parameters:
    - class_bitstrings (list of string): The class bit-strings
    """
    indexed_bitstrings = list(zip(range(len(class_bitstrings)), class_bitstrings))
    for i, bit_str_i in indexed_bitstrings:
        for j, bit_str_j in indexed_bitstrings[i:]:
            if i != j:
                hd = hamming_distance(bit_str_i, bit_str_j)
                print("Class {}, {} Hamming dist: {}".format(i, j, hd))


def hamming_distance(vector1, vector2):
    """
    Calculate the Hamming distance between two binary vectors
    
    Parameters:
    - vector1 (string): The first binary vector
    - vector2 (string): The second binary vector

    Returns:
    - h_dist (int): The Hamming distance between the two strings
    """
    h_dist = 0
    for v1, v2 in zip(vector1, vector2):
        if v1 != v2:
            h_dist += 1
    return h_dist