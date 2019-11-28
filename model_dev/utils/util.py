from typing import List 

def search(arr : List, global_index) -> int:
    """
    find closest greater or equal to index
    """
    global_index += 1  # change base 0 to base 1

    left = 0
    right = len(arr)-1
    while left<right:
        mid = (left + right) // 2
        if arr[mid]["length"] >= global_index:
            right = mid 
        elif arr[mid]["length"] < global_index:
            left = mid + 1

    result = (left + right) // 2
    return result

if __name__ == "__main__":
    global_index = 99 # 100 th element in base 0
    arr = [{"length":100}, {"length":150}, {"length":300},{"length":500}]
    result_index = search(arr, global_index)
    print(result_index)
