# NumPy Questions and Answers

1. **Q: Explain the difference between `np.array()` and `np.asarray()`. When would you use one over the other?**

   A: Both `np.array()` and `np.asarray()` can be used to create NumPy arrays, but they behave differently in certain scenarios:
   
   - `np.array()` always creates a new array.
   - `np.asarray()` creates a new array only if the input is not already a NumPy array with the same dtype. If the input is already a NumPy array with the same dtype, it returns the input array without copying.
   
   Use `np.asarray()` when you want to ensure you're working with a NumPy array but want to avoid unnecessary copying if the input is already a suitable array.

2. **Q: How can you create a NumPy array with a specific shape filled with a sequence of numbers?**

   A: You can use `np.arange()` combined with `reshape()`:

   ```python
   arr = np.arange(24).reshape(4, 6)
   ```

   This creates a 4x6 array filled with numbers from 0 to 23.

3. **Q: What's the difference between a shallow copy and a deep copy in NumPy? How can you create each?**

   A: 
   - A shallow copy creates a new array object but the elements still reference the same memory locations as the original array.
   - A deep copy creates a new array and recursively copies the elements.

   Create a shallow copy:
   ```python
   shallow_copy = original_array.view()
   ```

   Create a deep copy:
   ```python
   deep_copy = original_array.copy()
   ```

4. **Q: How can you find the unique rows in a 2D NumPy array?**

   A: You can use `np.unique()` with the `axis` parameter:

   ```python
   unique_rows = np.unique(arr, axis=0)
   ```

5. **Q: Explain broadcasting in NumPy and provide an example where it's useful.**

   A: Broadcasting is a mechanism that allows NumPy to perform operations on arrays of different shapes. The smaller array is "broadcast" across the larger array so that they have compatible shapes.

   Example:
   ```python
   arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   col_means = arr.mean(axis=0)
   centered_arr = arr - col_means
   ```

   Here, `col_means` is broadcast to match the shape of `arr`, allowing element-wise subtraction.

6. **Q: How can you efficiently compute the outer product of two 1D arrays?**

   A: Use `np.outer()`:

   ```python
   a = np.array([1, 2, 3])
   b = np.array([4, 5, 6])
   outer_product = np.outer(a, b)
   ```

7. **Q: What's the difference between `np.dot()` and `np.matmul()`? When would you use each?**

   A: 
   - `np.dot()` performs dot product of two arrays. For 2-D arrays, it's equivalent to matrix multiplication.
   - `np.matmul()` performs matrix product of two arrays.

   The main difference is in how they handle dimensions > 2:
   - `np.dot()` does not broadcast its arguments.
   - `np.matmul()` broadcasts its arguments.

   Use `np.dot()` for dot products and simple matrix multiplication. Use `np.matmul()` for more complex matrix operations, especially with higher dimensional arrays.

8. **Q: How can you create a NumPy array with random numbers from a specific probability distribution?**

   A: Use NumPy's random module. For example, to create an array with numbers from a normal distribution:

   ```python
   arr = np.random.normal(loc=0, scale=1, size=(3, 3))
   ```

9. **Q: Explain the concept of "fancy indexing" in NumPy and provide an example.**

   A: Fancy indexing allows you to select elements from an array using boolean arrays or integer arrays as indices.

   Example:
   ```python
   arr = np.arange(10)
   indices = [2, 5, 8]
   selected = arr[indices]  # selects elements at indices 2, 5, and 8
   ```

10. **Q: How can you efficiently compute the eigenvalues and eigenvectors of a square matrix using NumPy?**

    A: Use `np.linalg.eig()`:

    ```python
    matrix = np.array([[1, 2], [3, 4]])
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    ```

11. **Q: What's the difference between `np.save()` and `np.savez()`? When would you use each?**

    A: 
    - `np.save()` saves a single array to a file in .npy format.
    - `np.savez()` saves multiple arrays to a single file in .npz format.

    Use `np.save()` when you have a single array to save. Use `np.savez()` when you need to save multiple arrays or want to save arrays with their associated names.

12. **Q: How can you perform element-wise operations on two arrays with different shapes in NumPy?**

    A: You can use broadcasting if the arrays are compatible. If not, you might need to reshape one or both arrays. Example:

    ```python
    a = np.array([1, 2, 3])
    b = np.array([[1], [2], [3]])
    result = a + b  # b is broadcast to match the shape of a
    ```

13. **Q: Explain the concept of "structured arrays" in NumPy and provide an example of when they might be useful.**

    A: Structured arrays are arrays with structured datatypes. They allow you to create arrays with named fields of different types.

    Example:
    ```python
    dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    arr = np.array([('Alice', 25, 55.5), ('Bob', 30, 70.2)], dtype=dt)
    ```

    This is useful when you need to work with heterogeneous data, similar to a database table or a CSV file with different column types.

14. **Q: How can you efficiently compute the inverse of a matrix using NumPy?**

    A: Use `np.linalg.inv()`:

    ```python
    matrix = np.array([[1, 2], [3, 4]])
    inverse = np.linalg.inv(matrix)
    ```

15. **Q: What's the difference between `np.concatenate()` and `np.stack()`? When would you use each?**

    A: 
    - `np.concatenate()` joins a sequence of arrays along an existing axis.
    - `np.stack()` joins a sequence of arrays along a new axis.

    Use `np.concatenate()` when you want to join arrays along an existing dimension. Use `np.stack()` when you want to create a new dimension to stack the arrays.

16. **Q: How can you efficiently compute the determinant of a matrix using NumPy?**

    A: Use `np.linalg.det()`:

    ```python
    matrix = np.array([[1, 2], [3, 4]])
    determinant = np.linalg.det(matrix)
    ```

17. **Q: Explain the concept of "masked arrays" in NumPy and provide an example of when they might be useful.**

    A: Masked arrays are arrays that allow you to mark certain elements as invalid or missing. They're useful when dealing with datasets that have missing or invalid values.

    Example:
    ```python
    data = np.array([1, 2, -999, 4, 5])
    masked_data = np.ma.masked_array(data, mask=[False, False, True, False, False])
    ```

    This masks the value -999, which might represent a missing data point.

18. **Q: How can you efficiently solve a system of linear equations using NumPy?**

    A: Use `np.linalg.solve()`:

    ```python
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    x = np.linalg.solve(A, b)
    ```

    This solves the equation Ax = b for x.

19. **Q: What's the difference between `np.where()` and `np.select()`? When would you use each?**

    A: 
    - `np.where()` is used for simple conditional operations with two outcomes.
    - `np.select()` is used for more complex conditional operations with multiple conditions and choices.

    Use `np.where()` for simple if-else operations. Use `np.select()` when you have multiple conditions and corresponding values.

20. **Q: How can you efficiently compute the Fourier transform of a NumPy array?**

    A: Use `np.fft.fft()` for 1D transforms or `np.fft.fft2()` for 2D transforms:

    ```python
    arr = np.array([1, 2, 3, 4])
    fft_result = np.fft.fft(arr)
    ```

21. **Q: Explain the concept of "structured indexing" in NumPy and provide an example.**

    A: Structured indexing allows you to access fields of structured arrays using field names.

    Example:
    ```python
    dt = np.dtype([('name', 'U10'), ('age', 'i4')])
    arr = np.array([('Alice', 25), ('Bob', 30)], dtype=dt)
    ages = arr['age']
    ```

22. **Q: How can you efficiently compute the cross-product of two 3D vectors using NumPy?**

    A: Use `np.cross()`:

    ```python
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    cross_product = np.cross(v1, v2)
    ```

23. **Q: What's the difference between `np.einsum()` and traditional NumPy operations? When would you use `np.einsum()`?**

    A: `np.einsum()` provides a concise way to express many array operations using Einstein summation convention. It can often be more efficient and readable than combinations of other NumPy operations.

    Use `np.einsum()` for complex array operations that involve multiple axes and summations.

    Example (matrix multiplication):
    ```python
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    c = np.einsum('ij,jk->ik', a, b)
    ```

24. **Q: How can you efficiently compute the singular value decomposition (SVD) of a matrix using NumPy?**

    A: Use `np.linalg.svd()`:

    ```python
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    U, s, Vt = np.linalg.svd(matrix)
    ```

25. **Q: Explain the concept of "stride tricks" in NumPy and provide an example of when it might be useful.**

    A: Stride tricks allow you to create views of arrays with custom strides, enabling efficient operations on sliding windows or creating arrays with repeated elements.

    Example (creating a sliding window view):
    ```python
    from numpy.lib.stride_tricks import sliding_window_view
    arr = np.arange(10)
    windows = sliding_window_view(arr, window_shape=3)
    ```

26. **Q: How can you efficiently compute the correlation coefficient matrix for a set of variables using NumPy?**

    A: Use `np.corrcoef()`:

    ```python
    data = np.random.rand(100, 3)  # 100 samples, 3 variables
    correlation_matrix = np.corrcoef(data.T)
    ```

27. **Q: What's the difference between `np.vectorize()` and writing your own vectorized function? When would you use each?**

    A: `np.vectorize()` creates a vectorized function from a scalar function, but it's not always faster than a loop. Writing your own vectorized function using NumPy operations is often more efficient.

    Use `np.vectorize()` for quick prototyping or when performance isn't critical. Write your own vectorized function for optimal performance.

28. **Q: How can you efficiently compute the eigendecomposition of a symmetric matrix using NumPy?**

    A: Use `np.linalg.eigh()` for Hermitian (including real symmetric) matrices:

    ```python
    matrix = np.array([[1, 2], [2, 3]])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    ```

29. **Q: Explain the concept of "generalized ufuncs" in NumPy and provide an example.**

    A: Generalized ufuncs (universal functions) operate on whole sub-arrays rather than individual elements. They allow for more complex element-wise operations.

    Example:
    ```python
    def matmul_gufunc(a, b):
        return np.einsum('...ij,...jk->...ik', a, b)
    
    matmul = np.frompyfunc(matmul_gufunc, 2, 1)
    ```

    This creates a generalized ufunc for matrix multiplication.

30. **Q: How can you efficiently solve a sparse linear system using NumPy and SciPy?**

    A: Use SciPy's sparse module along with NumPy:

    ```python
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    import numpy as np

    A = sparse.csr_matrix([[1, 2], [3, 4]])
    b = np.array([5, 6])
    x = spsolve(A, b)
    ```

    This solves the sparse linear system Ax = b efficiently.