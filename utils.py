import numpy as np

def state_calculator(arr, low, mid):
    state_arr = []
    for i in arr:
        if i <= low:
            state_arr.append("L")
        elif i <= mid:
            state_arr.append("M")
        else:
            state_arr.append("H")
    return state_arr


def symbol_calculator(arr, a, b, c):
    output = []
    for i in arr:
        if i <= a:
            output.append("A")
        elif i <= b:
            output.append("B")
        elif i <= c:
            output.append("C")
        else:
            output.append("D")
    return output


def adjust_state_matrix(state_matrix):
    print("\n\nSum-----------------")
    # print(state_matrix)
    for i in range(27):
        if sum(state_matrix[i]) < 1:
            for j in range(27):
                state_matrix[i, j] = 1/27
        elif sum(state_matrix[i]) > 1:
            for j in range(27):
                state_matrix[i, j] = 1/27
    # print(state_matrix)
    print(np.sum(state_matrix, axis=1))
    return state_matrix
