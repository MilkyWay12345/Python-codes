def bubble_sort(nums):
    for i in range(len(nums)):
        for j in range(len(nums) - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]

if __name__ == '__main__':
    list_ = [5, 2, 1, 8, 4]
    bubble_sort(list_)
    print(list_)