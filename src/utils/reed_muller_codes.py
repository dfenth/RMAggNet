"""
An implementation of Reed-Muller codes with a brute-force correction method

Calculate basis sets with the recursive rules:
    - Hn includes a string with 2^(n-1) 0s followed by 2^(n-1) 1s
    - Includes two-fold repetitions of codewords from the H(n-1) basis set
"""

import itertools
import random, time

def generate_basis_set(n):
    """
    Generate a basis set of vectors, up to n instances

    Parameters:
    - n (int): The number of basis sets to generate

    Returns:
    (list of strings): A list of binary string representing the basis vectors
    """
    if n == 0:
        return "1"
    else:
        return ["0"*2**(n-1) + "1"*2**(n-1)] + [bit_str*2 for bit_str in generate_basis_set(n-1)]


def xor_bitstrings(s1, s2, str_len):
    """
    Performs the XOR operation on two binary strings

    Parameters:
    - s1 (string): The first binary string
    - s2 (string): The second binary string
    - str_len (int): The target length of the binary string (to keep a constant length since 0s can be cut from the start)

    Returns:
    - (string): The resulting binary string after XOR and zero padding
    """
    res = int(s1, base=2)^int(s2, base=2)
    return f"{res:0{str_len}b}"


def generate_codewords(basis):
    """
    Generate all codewords based on the basis set

    Parameters:
    - basis (list of string): A list of all basis binary strings

    Returns:
    - (list of string): All codewords possible from the basis set
    """
    # Can convert from a string to binary using bin(int("01010", base=2)) but we lose leading 0s
    combinations = basis_combinations(basis)
    codewords = []
    bit_len = len(basis[0])
    xor = lambda x,y: xor_bitstrings(x, y, bit_len)
    for c in combinations:
        bin_str = list(itertools.accumulate(c, xor, initial="0"*bit_len))[-1]
        codewords.append(bin_str)
    
    # Add in the 0 vector and concatenate with the basis and generated codewords
    return ["0"*bit_len]+basis+codewords 


def basis_combinations(basis):
    """
    Generates all possible combinations from a basis set

    Parameters:
    - basis (list of string): A list of basis vectors we need the combinations of

    Returns:
    - (list of (string, string)): All combinations of basis vectors
    """
    # Adapted from https://docs.python.org/3/library/itertools.html
    return itertools.chain.from_iterable(itertools.combinations(basis, n) for n in range(2,len(basis)+1))


def generate_reed_muller_sets(n, classes, seed=None):
    """
    Generate Reed-Muller codes of H_n
    
    Parameters:
    - n (int): The level of recursion of H
    - classes (list(int)): The list of classes to generate a code for
    - seed (int, optional): A random seed (default is None)
    
    Returns:
    - sets (list of list of int): The set of classes for each network
    - bitstr_list (list of string): The bitstrings associated with each class
    """
    basis = generate_basis_set(n)
    codewords = generate_codewords(basis)
    
    assert len(codewords) >= len(classes), "Error :: Not enough codewords for the number of classes"
    
    # Associate each class to a codeword
    class_to_bitstr = {}
    # Shuffle codewords to more evenly distribute the classes
    if not seed:
        seed = int(time.time())
    random.seed(seed)
    random.shuffle(codewords)

    # Remove '0'*n and '1'*n bitstrings
    tmp_codewords = []
    for idx, codeword in enumerate(codewords):
        if codeword != "0"*len(codeword) and codeword != "1"*len(codeword):
            tmp_codewords.append(codeword)
    codewords = tmp_codewords

    for i,c in enumerate(classes):
        class_to_bitstr[c] = codewords[i]

    # Create sets
    sets = []
    bit_len = len(codewords[0])
    for i in range(bit_len):
        tmp_set = []
        for key in list(class_to_bitstr):
            bit = class_to_bitstr[key][i]
            if bit == "1":
                tmp_set.append(key)

        sets.append(tmp_set)
    
    bitstr_list = [class_to_bitstr[x] for x in list(class_to_bitstr)]
    print(class_to_bitstr)
    print(sets)
    return sets, bitstr_list


# Example usage
if __name__ == "__main__":
    basis = generate_basis_set(5)
    print("Basis: {}".format(basis))

    # Basis combinations
    print("Basis combinations:")
    for b in basis_combinations(basis):
        print("\t",b)

    codewords = generate_codewords(basis)
    print("{} Codewords of length {}:".format(len(codewords), len(codewords[0])))
    for c in codewords:
        print("\t",c)

    generate_reed_muller_sets(4, [0,1,2,3,4,5,6,7,8,9])
