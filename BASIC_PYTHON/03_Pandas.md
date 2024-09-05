# Pandas Questions and Answers

1. **Q: What's the difference between `loc` and `iloc` in Pandas?**

   A: `loc` is label-based indexing, while `iloc` is integer position-based indexing. Use `loc` to select data based on its label and `iloc` to select data based on its integer position.

   ```python
   df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
   print(df.loc['b', 'A'])  # Returns 2
   print(df.iloc[1, 0])     # Also returns 2
   ```

2. **Q: How can you handle missing data in a Pandas DataFrame?**

   A: Use methods like `isnull()`, `dropna()`, `fillna()`, or `interpolate()`.

   ```python
   df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})
   df_filled = df.fillna(df.mean())
   ```

3. **Q: Explain the concept of "chaining" in Pandas and why it can be problematic.**

   A: Chaining applies multiple operations to a DataFrame in a single line. It can lead to unexpected behavior due to how Pandas handles copies and views.

   ```python
   # Problematic chaining
   df[df['A'] > 0]['B'] = 5  # This might not modify the original DataFrame

   # Better approach
   df.loc[df['A'] > 0, 'B'] = 5
   ```

4. **Q: How can you reshape a Pandas DataFrame using `melt` and `pivot`?**

   A: `melt` transforms from wide to long format, while `pivot` transforms from long to wide format.

   ```python
   # Melt example
   df_wide = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
   df_long = pd.melt(df_wide, var_name='Variable', value_name='Value')

   # Pivot example
   df_long = pd.DataFrame({'ID': [1, 1, 2, 2], 'Variable': ['A', 'B', 'A', 'B'], 'Value': [1, 2, 3, 4]})
   df_wide = df_long.pivot(index='ID', columns='Variable', values='Value')
   ```

5. **Q: What's the difference between `merge`, `join`, and `concat` in Pandas?**

   A: `merge` combines based on common columns/indexes, `join` combines based on indexes, and `concat` appends along an axis.

   ```python
   df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
   df2 = pd.DataFrame({'B': [3, 4], 'C': [5, 6]})
   
   merged = pd.merge(df1, df2, on='B')
   joined = df1.join(df2.set_index('B'), on='B')
   concatenated = pd.concat([df1, df2], axis=1)
   ```

6. **Q: How can you efficiently handle large datasets in Pandas that don't fit into memory?**

   A: Use `chunksize` in `read_csv`, use `dask` library, or use `SQLAlchemy` with databases.

   ```python
   chunks = pd.read_csv('large_file.csv', chunksize=10000)
   for chunk in chunks:
       process_data(chunk)
   ```

7. **Q: Explain the concept of MultiIndex in Pandas.**

   A: MultiIndex allows multiple levels of indexes on an axis, useful for higher-dimensional data.

   ```python
   arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
   df = pd.DataFrame(np.random.randn(8, 4), index=arrays)
   ```

8. **Q: How can you perform group-based operations in Pandas?**

   A: Use `groupby` with `apply`, `agg`, or `transform`.

   ```python
   df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'], 'B': [1, 2, 3, 4]})
   grouped = df.groupby('A')
   result = grouped.agg({'B': ['sum', 'mean']})
   ```

9. **Q: How can you handle time series data efficiently in Pandas?**

   A: Use `pd.to_datetime()`, `resample()`, `shift()`, and `rolling()`.

   ```python
   df = pd.DataFrame({'date': pd.date_range(start='2023-01-01', periods=5), 'value': range(5)})
   df.set_index('date', inplace=True)
   df_resampled = df.resample('2D').sum()
   ```

10. **Q: How can you optimize the performance of Pandas operations?**

    A: Use vectorized operations, appropriate data types, `inplace=True`, and consider `swifter` for parallelized operations.

    ```python
    df = pd.DataFrame({'A': np.random.choice(['a', 'b', 'c'], size=1000000)})
    df['A'] = df['A'].astype('category')  # Convert to categorical
    df['B'] = df['A'].map({'a': 1, 'b': 2, 'c': 3})  # Vectorized operation
    ```

11. **Q: What's the difference between `copy()` and `view()` in Pandas?**

    A: `copy()` creates a new DataFrame with copied data, while `view()` creates a new DataFrame object viewing the same data.

    ```python
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_copy = df.copy()
    df_view = df.view()
    ```

12. **Q: How can you handle outliers in a Pandas DataFrame?**

    A: Use statistical methods to identify outliers, then remove, cap, or transform them.

    ```python
    df = pd.DataFrame({'A': [1, 2, 3, 100, 5]})
    Q1 = df['A'].quantile(0.25)
    Q3 = df['A'].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df[(df['A'] >= Q1 - 1.5*IQR) & (df['A'] <= Q3 + 1.5*IQR)]
    ```

13. **Q: Explain the concept of categorical data in Pandas.**

    A: Categorical data is memory-efficient for columns with limited unique values.

    ```python
    df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue'] * 1000000})
    df['color'] = pd.Categorical(df['color'])
    ```

14. **Q: How can you create custom aggregation functions for group-by operations?**

    A: Define a custom function and use it with `agg()`.

    ```python
    def range_func(x):
        return x.max() - x.min()

    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'], 'B': [1, 2, 3, 4]})
    result = df.groupby('A').agg({'B': ['sum', 'mean', range_func]})
    ```

15. **Q: How can you handle duplicate data in a Pandas DataFrame?**

    A: Use `duplicated()` to identify duplicates and `drop_duplicates()` to remove them.

    ```python
    df = pd.DataFrame({'A': [1, 1, 2, 3, 3], 'B': [1, 1, 2, 3, 4]})
    df_unique = df.drop_duplicates(subset=['A'], keep='last')
    ```

16. **Q: What's the difference between `map()`, `apply()`, and `applymap()` in Pandas?**

    A: `map()` is for Series, `apply()` is for DataFrame axes, and `applymap()` is for every DataFrame element.

    ```python
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df['A'] = df['A'].map(lambda x: x*2)
    df['B'] = df['B'].apply(lambda x: x**2)
    df = df.applymap(lambda x: f'{x}!')
    ```

17. **Q: How can you efficiently compute rolling statistics in Pandas?**

    A: Use the `rolling()` method.

    ```python
    df = pd.DataFrame({'A': range(10)})
    df['rolling_mean'] = df['A'].rolling(window=3).mean()
    ```

18. **Q: Explain the concept of method chaining in Pandas.**

    A: Method chaining applies multiple methods to an object in a single line of code.

    ```python
    df = (pd.DataFrame({'A': range(10), 'B': range(10, 20)})
          .assign(C=lambda x: x['A'] + x['B'])
          .query('C > 15')
          .reset_index(drop=True))
    ```

19. **Q: How can you handle non-numeric data in Pandas when performing numerical operations?**

    A: Convert to numeric with `to_numeric()`, apply custom functions, or use one-hot encoding.

    ```python
    df = pd.DataFrame({'A': ['1', '2', '3', 'four'], 'B': [5, 6, 7, 8]})
    df['A'] = pd.to_numeric(df['A'], errors='coerce')
    df_encoded = pd.get_dummies(df, columns=['A'])
    ```

20. **Q: How can you efficiently compute pairwise correlations between columns in a large DataFrame?**

    A: Use the `corr()` method.

    ```python
    df = pd.DataFrame(np.random.randn(10000, 100))
    correlation_matrix = df.corr(method='pearson')
    ```

21. **Q: How can you handle hierarchical indexing in Pandas?**

    A: Use methods like `pd.MultiIndex.from_arrays()`, `loc`, `xs`, `unstack()`, and `stack()`.

    ```python
    arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
              np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]
    df = pd.DataFrame(np.random.randn(8, 4), index=arrays)
    ```

22. **Q: How can you create a Pandas DataFrame from a dictionary?**

    A: Use the `pd.DataFrame()` constructor with a dictionary.

    ```python
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)
    ```

23. **Q: How can you select rows in a DataFrame based on multiple conditions?**

    A: Use boolean indexing with & (and) or | (or) operators.

    ```python
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    result = df[(df['A'] > 2) & (df['B'] < 8)]
    ```

24. **Q: How can you rename columns in a Pandas DataFrame?**

    A: Use the `rename()` method or assign new column names directly.

    ```python
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df = df.rename(columns={'A': 'X', 'B': 'Y'})
    # or
    df.columns = ['X', 'Y']
    ```

25. **Q: How can you sort a DataFrame by multiple columns?**

    A: Use the `sort_values()` method with a list of column names.

    ```python
    df = pd.DataFrame({'A': [2, 1, 1, 2], 'B': [1, 2, 3, 4]})
    df_sorted = df.sort_values(['A', 'B'])
    ```

26. **Q: How can you create a new column based on values in other columns?**

    A: Use direct assignment or the `assign()` method.

    ```python
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df['C'] = df['A'] + df['B']
    # or
    df = df.assign(C=lambda x: x['A'] + x['B'])
    ```

27. **Q: How can you find the number of missing values in each column of a DataFrame?**

    A: Use `isnull()` or `isna()` with `sum()`.

    ```python
    df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    missing_counts = df.isnull().sum()
    ```

28. **Q: How can you convert a Pandas Series to a list?**

    A: Use the `tolist()` method or `list()` function.

    ```python
    s = pd.Series([1, 2, 3])
    lst = s.tolist()
    # or
    lst = list(s)
    ```

29. **Q: How can you group a DataFrame by one column and get the last N rows of each group?**

    A: Use `groupby()` with `tail()`.

    ```python
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
                       'B': range(6)})
    result = df.groupby('A').tail(2)
    ```

30. **Q: How can you convert a Pandas DataFrame to a NumPy array?**

    A: Use the `to_numpy()` method or `values` attribute.

    ```python
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    arr = df.to_numpy()
    # or
    arr = df.values
    ```
