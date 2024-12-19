import numpy as np
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
from itertools import product
import xml.etree.ElementTree as ET
from itertools import combinations, product
from datetime import timedelta
from tqdm import tqdm
import time
from joblib import Parallel, delayed

def ExactlyOne(vars):
    """Ensure exactly one of the variables in the list is True."""
    # At least one must be True
    at_least_one = Or(vars)
    # At most one must be True
    at_most_one = AtMost(*vars, 1)
    # Both conditions must be satisfied
    return And(at_least_one, at_most_one)


def one_entry_per_row(B):
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


def read_traces_from_xml(filename="state_traces.xml"):
    """
    Reads and parses the XML file containing state traces.

    Parameters:
    - filename: The XML file name to read from.

    Returns:
    - state_traces: A dictionary where keys are state indices and values are lists of grouped lists.
    """
    # Parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Initialize a dictionary to store the traces
    state_traces = {}

    # Iterate over each state element
    for state_element in root:
        state_id = int(state_element.tag.split("_")[1])  # Extract the state number
        state_traces[state_id] = []

        # Iterate over each list element under the state
        for list_element in state_element:
            # Split the text to get the group items back into a list
            group = list_element.text.split(", ")
            state_traces[state_id].append(group)

    return state_traces

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



if __name__ == '__main__':

    kappa = 4
    AP = 5
    total_variables = kappa**2*AP
    total_constraints = 0

    B = [[[Bool('x_%s_%s_%s'%(i,j,k) )for j in range(kappa)]for i in range(kappa)]for k in range(AP)]

    B_ = element_wise_or_boolean_matrices([b_k for b_k in B])
    x = [False]*kappa
    x[0] = True
    print(f"x = {x}")
    # for i, row in enumerate(B_):
    #     print(f"Row {i}: {[str(cell) for cell in row]}")
    
   
    B_T = transpose_boolean_matrix(B_)
    # for i, row in enumerate(B_T):
    #     print(f"B_T: Row {i}: {[str(cell) for cell in row]}")
    # print(boolean_matrix_vector_multiplication(boolean_matrix_power(B_T,2),x))

    powers_B_T = [boolean_matrix_power(B_T,k) for k in  range(1,kappa)]
    
    powers_B_T_x = [boolean_matrix_vector_multiplication(B,x) for B in powers_B_T]
    
    powers_B_T_x.insert(0, x)
    
    # print(powers_B_T_x[0])
    OR_powers_B_T_x = element_wise_or_boolean_vectors(powers_B_T_x)
    # print(OR_powers_B_T_x)
    s = Solver() # type: ignore

    # C0 Trace compression
    for ap in range(AP):
        for i in range(kappa):
            for j in range(kappa):
                # For boolean variables, B[ap][i][j], add the constraint that the current solution
                # is not equal to the previous solution
                s.add(Implies(B[ap][i][j], B[ap][j][j]))
                total_constraints +=1


    # C1 and C2 from Notion Write-up
    for k in range(AP):
        total_constraints +=1
        s.add(one_entry_per_row(B[k]))

    # # C3 from from Notion Write-up
    # for element in OR_powers_B_T_x:
    #     total_constraints +=1
    #     s.add(element)
    

    proposition2index = {'A':0, 'B':1 , 'C':2, 'D':3, 'H':4}

    def prefix2indices(s):
        out = []
        for l in s:
            out.append(proposition2index[l])
        return out

    # n_counter = 1

    # counter_examples = {}
    # for ce in range(n_counter):
    #     if ce not in counter_examples.keys():
    #         counter_examples[ce] = []

    # I will hard code the counter examples for
    # SET1 = ['A', 'AA', 'AAA', 'BA', 'BAA', 'BBA']
    # SET2 = ['B', 'BB', 'BBB','ABB']
    # SET3 = ['AB','AAB','BAB']
    # SET4 = ['ABA']

    # Read traces from XML
    parsed_traces = read_traces_from_xml()

    # Display the parsed traces
    # print("\nParsed Traces from XML:")
    # for state, lists in parsed_traces.items():
    #     print(f"State {state}:")
    #     for i, lst in enumerate(lists, 1):
    #         print(f"  l{i} = {lst}")

    # Generate cross products for each state
    counter_examples = generate_combinations(parsed_traces)



    # C4 from from Notion Write-up 
    print("Started with C4 ... \n")
    total_start_time = time.time()

    ce_set = [('B', 'AB'),
        ('ABA', 'A'),
        ('ABCB', 'B'),
        ('ABCB', 'AB'),
        ('ABCDC', 'ADC'),
        ('ABCDC', 'ABC'),
        ('CB', 'AB'),
        ('ABC', 'C'),
        ('BC', 'ABC'),
        ('B', 'ABCB'),
        ('ABC', 'AC'),
        ('C', 'AC'),
        ('ABCD', 'AD'),
        ('ABC', 'ADC'),
        ('ABAD', 'ABCD'),
        ('ABCDA', 'ABA'),
        ('ABDA', 'A'),
        ('ABD', 'AD'),
        ('ABCB', 'AB'),
        ('ABCDA', 'ABCBA'),
        ('ABCA', 'A'),
        ('ABCA', 'ABA')]

    # for ce in tqdm(ce_set,desc="Processing Counterexamples"):
    #     p1 = prefix2indices(ce[0])
    #     p2 = prefix2indices(ce[1])

    #     # Now
    #     sub_B1 = bool_matrix_mult_from_indices(B,p1, x)
    #     sub_B2 = bool_matrix_mult_from_indices(B,p2, x)

    #     res_ = element_wise_and_boolean_vectors(sub_B1, sub_B2)

    #     for elt in res_:
    #         s.add(Not(elt))



    n_counter = 9
    for state in range(n_counter):
        print(f"Currently in state {state}...")
        ce_set = counter_examples[state]
        print(f"The number of counter examples is: {len(ce_set)}\n")
        total_constraints += len(ce_set)
        
        # for each counter example in this set, add the correspodning constraint
        for ce in tqdm(ce_set,desc="Processing Counterexamples"):
            p1 = prefix2indices(ce[0])
            p2 = prefix2indices(ce[1])

            # Now
            sub_B1 = bool_matrix_mult_from_indices(B,p1, x)
            sub_B2 = bool_matrix_mult_from_indices(B,p2, x)

            res_ = element_wise_and_boolean_vectors(sub_B1, sub_B2)

            for elt in res_:
                s.add(Not(elt))
                
        
    print(f"we have a tortal of {total_constraints}!")
    # Use timedelta to format the elapsed time
    elapsed  = time.time() - total_start_time
    formatted_time = str(timedelta(seconds= elapsed))

    # Add milliseconds separately
    milliseconds = int((elapsed % 1) * 1000)

    # Format the time string
    formatted_time = formatted_time.split('.')[0] + f":{milliseconds:03}"
    print(f"Adding C4 took {formatted_time} seconds.")

    # # # no
   
    import time
    # start = time.time()
    # s.check()
    # end = time.time()
    # print(f"The SAT solver took {end - start} seconds to solve a problem with {total_variables} variables and >= {total_constraints*kappa} constraints.")
    # if s.check() == sat:
    #     print("Yup!")
    # else:
    #     print("NOT SAT")
    # m = s.model()

    # for ap in range(AP):
    #     r = [[ m.evaluate(B[ap][i][j]) for j in range(kappa)] for i in range(kappa)]
    #     print_matrix(r)

    # Record all solutions
    # solutions = []

    # Start the timer
    start = time.time()
    s_it = 0
    while True:
        # Solve the problem
        if s.check() == sat:
            end = time.time()
            print(f"The SAT solver took: {end-start} sec.")
            # Get the current solution
            m = s.model()
            
            # # Store the current solution
            # solution = []
            print(f"Solution {s_it} ...")
            for ap in range(AP):
                r = [[m.evaluate(B[ap][i][j]) for j in range(kappa)] for i in range(kappa)]
                # solution.append(r)
                
                print_matrix(r)  # Assuming print_matrix prints your matrix nicely
            s_it += 1
            # # Add the solution to the list of found solutions
            # solutions.append(solution)

            # Build a clause that ensures the next solution is different
            # The clause is essentially that at least one variable must differ
            block_clause = []
            for ap in range(AP):
                for i in range(kappa):
                    for j in range(kappa):
                        # For boolean variables, B[ap][i][j], add the constraint that the current solution
                        # is not equal to the previous solution
                        block_clause.append(B[ap][i][j] != m.evaluate(B[ap][i][j], model_completion=True))

            # Add the blocking clause to the solver
            s.add(Or(block_clause))
            
        else:
            print("NOT SAT - No more solutions!")
            break