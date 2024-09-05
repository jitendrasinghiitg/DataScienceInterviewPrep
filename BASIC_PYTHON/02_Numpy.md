# Advanced NumPy Concept Clearing Questions

1. **What is broadcasting in NumPy, and how does it work?**
   Broadcasting is a method that allows NumPy to work with arrays of different shapes when performing arithmetic operations. It involves expanding one or both arrays so that they have compatible shapes for element-wise operations.

2. **How does NumPy handle multidimensional arrays in terms of memory layout?**
   NumPy uses a contiguous block of memory to store multidimensional arrays. The elements are stored in a row-major (C-style) or column-major (Fortran-style) order.

3. **Explain the concept of vectorization in NumPy.**
   Vectorization refers to the practice of replacing explicit loops with array expressions to make the code more concise and efficient. NumPy achieves vectorization through operations that are implemented in C.

4. **What are ufuncs in NumPy?**
   Ufuncs, or universal functions, are functions that operate element-wise on arrays. They are implemented in C for performance and support broadcasting, typecasting, and other features.

5. **How do you perform element-wise addition of two arrays with different shapes using broadcasting?**
   Arrays must have compatible shapes for broadcasting. For example, adding an array of shape (3, 1) to an array of shape (1, 4) results in an array of shape (3, 4).

6. **What is the difference between np.dot and np.matmul?**
   `np.dot` is used for dot product of two arrays, while `np.matmul` is used for matrix multiplication, supporting broadcasting rules.

7. **How can you invert a matrix using NumPy?**
   Use `np.linalg.inv()` to invert a square matrix.

8. **What is the role of the axis parameter in NumPy functions?**
   The axis parameter specifies the axis along which an operation is performed. For example, summing along axis 0 sums the columns, and summing along axis 1 sums the rows.

9. **How do you handle missing data in NumPy arrays?**
   Use masked arrays from the `numpy.ma` module or replace missing values with `np.nan` and use functions like `np.nanmean` to handle them.

10. **What are structured arrays in NumPy?**
    Structured arrays are ndarrays with a structured dtype. They allow each element to be a record, containing multiple named fields of potentially different types.

11. **How can you concatenate multiple arrays along a specific axis?**
    Use `np.concatenate` and specify the axis parameter.

12. **How do you find the eigenvalues and eigenvectors of a matrix using NumPy?**
    Use `np.linalg.eig()` to compute eigenvalues and eigenvectors.

13. **What is the difference between np.array and np.asarray?**
    `np.array` always creates a new array, while `np.asarray` converts an input to an array only if it is not already an ndarray.

14. **How can you solve a system of linear equations using NumPy?**
    Use `np.linalg.solve()` to solve a linear matrix equation.

15. **What is the purpose of np.meshgrid?**
    `np.meshgrid` generates coordinate matrices from coordinate vectors, useful for evaluating functions on a grid.

16. **How do you compute the inverse of a matrix using NumPy?**
    Use `np.linalg.inv()` to compute the inverse of a square matrix.

17. **How can you flatten a NumPy array?**
    Use the `flatten()` method to return a copy of the array collapsed into one dimension, or `ravel()` for a flattened view.

18. **What is the difference between np.sum and np.add.reduce?**
    `np.sum` computes the sum of array elements along a given axis, while `np.add.reduce` performs a reduction using the addition operator.

19. **How do you perform element-wise multiplication of two arrays?**
    Use the `*` operator or `np.multiply()` for element-wise multiplication.

20. **What are views in NumPy, and how are they different from copies?**
    Views are arrays that share the same data as the original array but can be reshaped or sliced differently. Copies are new arrays with their own data.

21. **How can you generate random numbers using NumPy?**
    Use functions from the `np.random` module, such as `np.random.rand` for uniform distribution or `np.random.randn` for normal distribution.

22. **What is the use of np.linalg.qr()?**
    `np.linalg.qr()` computes the QR decomposition of a matrix.

23. **How do you compute the dot product of two vectors in NumPy?**
    Use `np.dot()` to compute the dot product of two arrays.

24. **What is the purpose of np.vectorize()?**
    `np.vectorize()` is a convenience function for vectorizing functions that do not natively support NumPy arrays.

25. **How do you compute the cumulative sum of elements in a NumPy array?**
    Use `np.cumsum()` to compute the cumulative sum along a given axis.

26. **What is the difference between np.where() and np.nonzero()?**
    `np.where()` returns elements chosen from `x` or `y` depending on the condition, while `np.nonzero()` returns the indices of non-zero elements.

27. **How can you reshape a NumPy array?**
    Use the `reshape()` method to change the shape of an array without changing its data.

28. **What are fancy indexing and slicing in NumPy?**
    Fancy indexing involves passing an array of indices to access multiple array elements, while slicing is used to access a range of elements.

29. **How do you compute the mean of an array along a specific axis?**
    Use `np.mean()` and specify the axis parameter.

30. **What is the difference between np.linalg.svd() and np.linalg.eig()?**
    `np.linalg.svd()` performs singular value decomposition, while `np.linalg.eig()` computes eigenvalues and eigenvectors.

31. **How do you sort an array along a specific axis?**
    Use `np.sort()` and specify the axis parameter.

32. **What is the purpose of np.histogram()?**
    `np.histogram()` computes the histogram of a set of data, returning the bin counts and bin edges.

33. **How do you find unique elements in an array?**
    Use `np.unique()` to find unique elements in an array.

34. **What is the difference between np.copy() and the copy method of a NumPy array?**
    Both create a new array that is a copy of the original array, but `np.copy()` is a function, while `copy()` is a method of the ndarray.

35. **How can you stack arrays vertically or horizontally?**
    Use `np.vstack()` to stack arrays vertically and `np.hstack()` to stack arrays horizontally.

36. **What are the benefits of using np.may_share_memory()?**
    `np.may_share_memory()` checks if two arrays might share memory, which is useful for optimizing performance.

37. **How do you compute the covariance matrix of an array?**
    Use `np.cov()` to compute the covariance matrix.

38. **What is the difference between np.array_equal() and np.array_equiv()?**
    `np.array_equal()` checks if two arrays have the same shape and elements, while `np.array_equiv()` allows broadcasting before comparison.

39. **How do you find the indices of the maximum value in an array?**
    Use `np.argmax()` to find the indices of the maximum value.

40. **How can you compute the inner product of two arrays?**
    Use `np.inner()` to compute the inner product.

41. **What is the purpose of np.nditer()?**
    `np.nditer()` provides an efficient multi-dimensional iterator object to iterate over arrays.

42. **How do you split an array into multiple sub-arrays?**
    Use `np.split()` to split an array into multiple sub-arrays.

43. **What are the benefits of using np.put() and np.take()?**
    `np.put()` allows placing values into an array at specified indices, while `np.take()` allows extracting elements from an array at specified indices.

44. **How can you round elements of an array to the nearest integer?**
    Use `np.round()` to round elements to the nearest integer.

45. **What is the use of np.ptp()?**
    `np.ptp()` returns the range (maximum - minimum) of values along an axis.

46. **How do you compute the cross product of two vectors?**
    Use `np.cross()` to compute the cross product.

47. **What is the purpose of np.meshgrid()?**
    `np.meshgrid()` creates coordinate matrices from coordinate vectors, useful for evaluating functions on a grid.

48. **How can you create a diagonal matrix using NumPy?**
    Use `np.diag()` to create a diagonal matrix.

49. **What is the difference between np.real() and np.imag()?**
    `np.real()` returns the real part of a complex array, while `np.imag()` returns the imaginary part.

50. **How do you compute the standard deviation of an array?**
    Use `np.std()` to compute the standard deviation along a specified axis.
