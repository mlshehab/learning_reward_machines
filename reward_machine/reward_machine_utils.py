"""
This work includes code derived from the "reward_machines" project 
(https://github.com/RodrigoToroIcarte/reward_machines)
"""

# def evaluate_dnf(formula, true_props):
#     """
#     Evaluates a formula in Disjunctive Normal Form (DNF) given a set of true propositions.
    
#     Parameters:
#         formula (str): The DNF formula as a string, where conjunctions (AND) are '&' and 
#                        disjunctions (OR) are '|'. Negations are denoted by '!'.
#         true_props (set): A set of propositions that are true.
    
#     Returns:
#         bool: True if the formula evaluates to True with the given true_props, False otherwise.
#     """
#     # Split the formula into disjunctions (OR parts)
#     disjunctions = formula.split('|')
    
#     # Check each disjunction
#     for conj in disjunctions:
#         # Split the conjunction (AND parts)
#         literals = conj.split('&')
#         conj_true = True
#         for literal in literals:
#             literal = literal.strip()
#             if literal.startswith('!'):
#                 # Negated literal
#                 prop = literal[1:]
#                 if prop in true_props:  # Negated literal should not be in true_props
#                     conj_true = False
#                     break
#             else:
#                 # Positive literal
#                 if literal not in true_props:  # Literal should be in true_props
#                     conj_true = False
#                     break
#         if conj_true:
#             # If any conjunction is true, the formula is true
#             return True
    
#     # If no conjunction is true, the formula is false
#     return False


def evaluate_dnf(formula, true_props):
    """
    DNF stands for Disjunctive Normal Form. It is a standard way of structuring a logical 
    formula in Boolean algebra. A formula in DNF is a disjunction (OR) of conjunctions (AND)
     of literals, where a literal is either a variable or its negation.
    """
    # Split the formula into conjunctions
    conjunctions = formula.split('|')
    # print(f"The conju are: {conjunctions}")
    # Check each conjunction
    for conj in conjunctions:
        literals = conj.split('&')
        conj_true = True
        for literal in literals:
            literal = literal.strip()
            if literal.startswith('!'):
                # Negated literal
                prop = literal[1:]
                if prop in true_props:
                    conj_true = False
                    break
            else:
                # Positive literal
                if literal not in true_props:
                    conj_true = False
                    break
        if conj_true:
         
            return True
    return False


def value_iteration(U, delta_u, delta_r, terminal_u, gamma):
    """
    Standard value iteration approach. 
    We use it to compute the potential function for the automated reward shaping
    """
    V = dict([(u,0) for u in U])
    V[terminal_u] = 0
    V_error = 1
    while V_error > 0.0000001:
        V_error = 0
        for u1 in U:
            q_u2 = []
            for u2 in delta_u[u1]:
                if delta_r[u1][u2].get_type() == "constant": 
                    r = delta_r[u1][u2].get_reward(None)
                else:
                    r = 0 # If the reward function is not constant, we assume it returns a reward of zero
                q_u2.append(r+gamma*V[u2])
            v_new = max(q_u2)
            V_error = max([V_error, abs(v_new-V[u1])])
            V[u1] = v_new
    return V

if __name__ == '__main__':
    
    # Example usage:
    print(evaluate_dnf("a&b|!c&d", "b"))  # Output: True