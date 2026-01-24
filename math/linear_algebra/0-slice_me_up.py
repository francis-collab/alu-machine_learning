#!/usr/bin/env python3

"""Module for slicing lists."""

# Define the original array
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]

# Slice the first two elements
arr1 = arr[:2]

# Slice the last five elements
arr2 = arr[-5:]

# Slice from the 2nd (index 1) to the 6th (index 5) elements
arr3 = arr[1:6]

# Print the results as specified
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
