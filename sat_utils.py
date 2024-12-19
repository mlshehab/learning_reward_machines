from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
import numpy as np
from itertools import combinations, product





def ExactlyOne(vars):
    """Ensure exactly one of the variables in the list is True."""
    # At least one must be True
    at_least_one = Or(vars)
    # At most one must be True
    at_most_one = AtMost(*vars, 1)
    # Both conditions must be satisfied
    return And(at_least_one, at_most_one)


def one_entry_per_row(B):
    # print(f"The shape of B is: {len(B)}")
    kappa = len(B)
    cond = []
    for i in range(kappa):
        cond+= [ExactlyOne([B[i][j] for j in range(kappa)])]
    return cond

def boolean_matrix_vector_multiplication(A,b):
    # def boolean_matrix_vector_multiplication(matrix, vector):
    # Number of rows in matrix
    num_rows = len(A)
    # Number of columns in matrix (assuming non-empty matrix)
    num_cols = len(A[0])
    # print(f"The numerb of cols is {num_cols}")
    # Ensure the vector size matches the number of columns in the matrix
    assert len(b) == num_cols

    # Resulting vector after multiplication
    result = []

    # Perform multiplication
    for i in range(num_rows):
        # For each row in the matrix, compute the result using AND/OR operations
        # result_i = OR(AND(matrix[i][j], vector[j]) for all j)
        row_result = Or([And(A[i][j], b[j]) for j in range(num_cols)])
        result.append(row_result)
    
    return result


# Function for matrix-matrix boolean multiplication
def boolean_matrix_matrix_multiplication(A, B):
    # Number of rows in matrix A and columns in matrix B
    num_rows_A = len(A)
    num_cols_B = len(B[0])
    
    # Number of columns in A and rows in B (must match for matrix multiplication)
    num_cols_A = len(A[0])
    num_rows_B = len(B)
    assert num_cols_A == num_rows_B, "The number of columns in A must equal the number of rows in B."
    
    # Resulting matrix after multiplication
    result = [[None for _ in range(num_cols_B)] for _ in range(num_rows_A)]

    # Perform multiplication
    for i in range(num_rows_A):
        for j in range(num_cols_B):
            # Compute C[i][j] = OR(AND(A[i][k], B[k][j]) for all k)
            result[i][j] = Or([And(A[i][k], B[k][j]) for k in range(num_cols_A)])
    
    return result


def transpose_boolean_matrix(matrix):
    # Number of rows and columns in the input matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Initialize the transposed matrix
    transposed = [[None for _ in range(num_rows)] for _ in range(num_cols)]

    # Transpose operation: Swap rows and columns
    for i in range(num_rows):
        for j in range(num_cols):
            transposed[j][i] = matrix[i][j]
    
    return transposed

# Function to compute the kth power of a boolean matrix
def boolean_matrix_power(matrix, k):
    # Get the size of the matrix (assuming square matrix)
    n = len(matrix)
    assert all(len(row) == n for row in matrix), "The input matrix must be square."

    # Initialize the result matrix as the input matrix (A^1)
    
    result = matrix
    if k == 0 or k == 1:
        return result

    # Multiply the matrix by itself k-1 times
    for _ in range(k - 1):
        result = boolean_matrix_matrix_multiplication(result, matrix)
    
    return result

# Function to compute the element-wise OR of a list of boolean matrices
def element_wise_or_boolean_matrices(matrices):
    # Ensure there is at least one matrix
    assert len(matrices) > 0, "There must be at least one matrix in the list."

    # Get the number of rows and columns from the first matrix
    num_rows = len(matrices[0])
    num_cols = len(matrices[0][0])

    # Ensure all matrices have the same dimensions
    for matrix in matrices:
        assert len(matrix) == num_rows and all(len(row) == num_cols for row in matrix), "All matrices must have the same dimensions."

    # Initialize the result matrix
    result = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    # Compute the element-wise OR for each element
    for i in range(num_rows):
        for j in range(num_cols):
            # OR all matrices at position (i, j)
            result[i][j] = Or([matrix[i][j] for matrix in matrices])
    
    return result

# Function to compute the element-wise OR of a list of boolean vectors
def element_wise_or_boolean_vectors(vectors):
    # Ensure there is at least one vector
    assert len(vectors) > 0, "There must be at least one vector in the list."

    # Get the length of the first vector
    vector_length = len(vectors[0])

    # Ensure all vectors have the same length
    for vector in vectors:
        assert len(vector) == vector_length, "All vectors must have the same length."

    # Initialize the result vector
    result = [None] * vector_length

    # Compute the element-wise OR for each element
    for i in range(vector_length):
        # OR all vectors at position i
        result[i] = Or([vector[i] for vector in vectors])
    
    return result

def bool_matrix_mult_from_indices(B,indices, x):
    # indices = [l0, l1 , l2 ,... , l_k]
    # Get the number of rows and columns from the first matrix
    num_rows = len(B[0])
    num_cols = len(B[0][0])
    # print(f"The B[0] matrix is of shape {num_rows} by {num_cols}")

    len_trace = len(indices)

    result = transpose_boolean_matrix(B[indices[0]])
    
    i = 0
    for i in range(1,len_trace):
        # print(f"i = {i}, len_Trace = {len_trace}")
        result = boolean_matrix_matrix_multiplication(transpose_boolean_matrix(B[indices[i]]), result)
        
    return boolean_matrix_vector_multiplication(result,x)

def element_wise_and_boolean_vectors(vector1, vector2):
    # Ensure both vectors have the same length
    assert len(vector1) == len(vector2), "Both vectors must have the same length."

    # Initialize the result vector
    result = [None] * len(vector1)

    # Compute the element-wise AND for each element
    for i in range(len(vector1)):
        # AND the corresponding elements from both vectors
        result[i] = And(vector1[i], vector2[i])
    
    return result


def generate_combinations(traces_dict):
    
    combinations_dict = {}

    for state, lists in traces_dict.items():
        all_combinations = []
        # Generate combinations for different lengths of lists
        r = 2
        for combination in combinations(lists, r):
            # print(f"Generating combinations for state {state}, combination size {r}: {combination}")
            cross_products = list(product(*combination))
            all_combinations.extend(cross_products)
        
        combinations_dict[state] = all_combinations

    return combinations_dict
