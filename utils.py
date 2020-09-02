import pandas as pd


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


def export_csv(data, filename, output_path):
    d = {}
    for i in data:
        d[i] = i
    df = pd.DataFrame(data=d)
    df.to_csv(f"{output_path}//{filename}")
    print("Csv file created")
