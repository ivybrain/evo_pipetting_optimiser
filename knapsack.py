import numpy as np


def knapsack(sum, arr):

    n = len(arr)
    table = np.zeros((n + 1, sum + 1))

    for i in range(1, n + 1):
        for j in range(1, sum + 1):
            if j - arr[i - 1][0] >= 0:
                table[i, j] = max(
                    table[i - 1, j], table[i - 1, j - arr[i - 1][0]] + arr[i - 1][1]
                )
            else:
                table[i, j] = table[i - 1, j]

    print(table)

    include_counter = n
    included_elems = []
    sum_accounted = sum
    while include_counter > 0:
        # Subset is still achievable without this element, so it's not included
        if (
            table[include_counter, sum_accounted]
            == table[include_counter - 1, sum_accounted]
        ):
            include_counter -= 1
        else:
            included_elems.append(include_counter - 1)
            sum_accounted -= arr[include_counter - 1][0]
            include_counter -= 1

    return included_elems


# Input
# Tuples of weight, profit
arr = [(1, 2), (2, 2), (3, 3), (7, 4), (8, 3), (10, 5)]
print("Dynamic Programming Solution:", knapsack(11, arr))
