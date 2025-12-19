def right_bound(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    if right < 0:
        return right
    return left

def left_bound(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            right = mid - 1
    if right < 0:
        return right
    return left

array = [3, 7, 19, 24, 25, 31, 33, 40, 47]

print(left_bound(array, 33))
print(right_bound(array, 33))