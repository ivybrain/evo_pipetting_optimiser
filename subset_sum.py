import numpy as np


def subset_sum(sum, arr):

    n = len(arr)
    table = np.full((n + 1, sum + 1), False)
    # Any sum of size 0 is reachable
    table[:, 0] = True

    for i in range(1, n + 1):
        for j in range(1, sum + 1):
            if arr[i - 1] > j:
                table[i, j] = table[i - 1, j]
            else:
                table[i, j] = table[i - 1, j] or table[i - 1, j - arr[i - 1]]

    print(table)

    if not table[n, sum]:
        return False

    include_counter = n
    included_elems = []
    sum_accounted = sum
    while sum_accounted > 0:
        # Subset is still achievable without this element, so it's not included
        if table[include_counter - 1, sum_accounted]:
            include_counter -= 1
        else:
            included_elems.append(include_counter - 1)
            sum_accounted -= arr[include_counter - 1]
            include_counter -= 1

    return included_elems


# Input
arr = [2, 3, 7, 8, 10]
print("Dynamic Programming Solution:", subset_sum(11, arr))
