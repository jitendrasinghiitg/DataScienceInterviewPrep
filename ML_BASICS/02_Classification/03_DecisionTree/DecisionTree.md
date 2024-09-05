# Decision Trees Concept Clearing Questions

### Algorithm

1. **What is a Decision Tree?**
   **Answer:** A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It splits the data into subsets based on the value of input features to make predictions.

2. **How does a Decision Tree work?**
   **Answer:** A Decision Tree works by recursively splitting the data into subsets based on the feature that results in the most significant information gain or the smallest impurity, creating a tree-like structure of decisions.

3. **What is the difference between a classification tree and a regression tree?**
   **Answer:** A classification tree is used for predicting categorical outcomes, while a regression tree is used for predicting continuous outcomes.

4. **What is the purpose of the root node in a Decision Tree?**
   **Answer:** The root node is the topmost node in a Decision Tree, representing the entire dataset. It is the starting point for the splitting process.

5. **What are internal nodes and leaf nodes in a Decision Tree?**
   **Answer:** Internal nodes represent decisions based on a feature and lead to further splits. Leaf nodes represent the final outcome or prediction.

6. **What is a decision stump?**
   **Answer:** A decision stump is a Decision Tree with only one internal node (root node) and two leaf nodes. It makes decisions based on a single feature.

7. **What is the Gini impurity, and how is it used in Decision Trees?**
   **Answer:** Gini impurity measures the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the distribution of labels in the dataset. It is used to determine the best feature to split on.

8. **What is entropy, and how is it used in Decision Trees?**
   **Answer:** Entropy measures the disorder or uncertainty in a dataset. In Decision Trees, it is used to calculate information gain, which determines the best feature to split on.

9. **What is information gain, and how is it calculated?**
   **Answer:** Information gain is the reduction in entropy or impurity achieved by partitioning the data based on a feature. It is calculated as the difference between the entropy of the dataset before and after the split.

10. **What is the CART algorithm?**
    **Answer:** The CART (Classification and Regression Trees) algorithm is a popular method for constructing Decision Trees. It uses the Gini impurity for classification tasks and the mean squared error for regression tasks.

### Implementation

11. **How do you implement a Decision Tree in Python using scikit-learn?**
    **Answer:** You can implement a Decision Tree using the `DecisionTreeClassifier` or `DecisionTreeRegressor` class in scikit-learn. Example:
    ```python
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ```

12. **What are the key hyperparameters to tune in a Decision Tree model?**
    **Answer:** Key hyperparameters include `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `criterion`.

13. **What is the effect of the `max_depth` parameter in a Decision Tree?**
    **Answer:** The `max_depth` parameter limits the depth of the tree, preventing it from growing too deep and overfitting the training data.

14. **How do you handle missing values in Decision Trees?**
    **Answer:** Decision Trees can handle missing values by using surrogate splits or by imputing missing values before training the model.

15. **What is the role of the `min_samples_split` parameter in a Decision Tree?**
    **Answer:** The `min_samples_split` parameter specifies the minimum number of samples required to split an internal node, preventing the tree from creating splits that are not statistically significant.

16. **What is the role of the `min_samples_leaf` parameter in a Decision Tree?**
    **Answer:** The `min_samples_leaf` parameter specifies the minimum number of samples required to be at a leaf node, ensuring that leaf nodes are not created with too few samples.

17. **How do you visualize a Decision Tree in Python using scikit-learn?**
    **Answer:** You can visualize a Decision Tree using the `plot_tree` function in scikit-learn. Example:
    ```python
    from sklearn.tree import plot_tree
    plot_tree(model)
    ```

18. **What is the `max_features` parameter in a Decision Tree, and how does it affect the model?**
    **Answer:** The `max_features` parameter specifies the maximum number of features to consider when looking for the best split. It can help reduce overfitting by limiting the number of features used.

19. **How do you export a Decision Tree to a graphical representation?**
    **Answer:** You can export a Decision Tree to a graphical representation using the `export_graphviz` function in scikit-learn. Example:
    ```python
    from sklearn.tree import export_graphviz
    export_graphviz(model, out_file='tree.dot')
    ```

20. **What is the role of the `criterion` parameter in a Decision Tree?**
    **Answer:** The `criterion` parameter specifies the function used to measure the quality of a split. Common criteria include `gini` for Gini impurity and `entropy` for information gain.

### Hyperparameters

21. **What is grid search in the context of Decision Trees?**
    **Answer:** Grid search is a hyperparameter tuning technique that exhaustively searches over a specified parameter grid to find the optimal hyperparameter values for the model.

22. **How do you perform grid search for hyperparameter tuning in Decision Trees using scikit-learn?**
    **Answer:** You can use the `GridSearchCV` class in scikit-learn. Example:
    ```python
    from sklearn.model_selection import GridSearchCV
    parameters = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    model = DecisionTreeClassifier()
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    ```

23. **What is cross-validation and why is it important in Decision Trees?**
    **Answer:** Cross-validation is a technique to assess the generalizability of a model by splitting the data into training and validation sets multiple times. It helps prevent overfitting and provides a more reliable estimate of model performance.

24. **How do you perform cross-validation in scikit-learn for Decision Trees?**
    **Answer:** Use the `cross_val_score` function in scikit-learn. Example:
    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
    ```

25. **What is the purpose of using randomized search for hyperparameter tuning in Decision Trees?**
    **Answer:** Randomized search randomly samples hyperparameter values from a specified distribution, allowing a more efficient search compared to grid search, especially when the hyperparameter space is large.

26. **How do you perform randomized search for hyperparameter tuning in Decision Trees using scikit-learn?**
    **Answer:** You can use the `RandomizedSearchCV` class in scikit-learn. Example:
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    parameters = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    model = DecisionTreeClassifier()
    clf = RandomizedSearchCV(model, parameters, n_iter=10)
    clf.fit(X_train, y_train)
    ```

27. **What is the effect of the `max_leaf_nodes` parameter in a Decision Tree?**
    **Answer:** The `max_leaf_nodes` parameter limits the number of leaf nodes in the tree, which can help prevent overfitting by simplifying the tree structure.

28. **How do you interpret the feature importances in a Decision Tree?**
    **Answer:** Feature importances indicate the contribution of each feature to the model's predictions. Higher values indicate more important features. They can be accessed using the `feature_importances_` attribute.

29. **What is the effect of the `min_weight_fraction_leaf` parameter in a Decision Tree?**
    **Answer:** The `min_weight_fraction_leaf` parameter specifies the minimum weighted fraction of the input samples required to be at a leaf node, ensuring that leaf nodes are not created with too few samples, especially in cases with imbalanced classes.

30. **What is the role of the `splitter` parameter in a Decision Tree?**
    **Answer:** The `splitter` parameter specifies the strategy used to choose the split at each node. Options include `best` for the best split and `random` for a random split.

### Evaluation

31. **How do you evaluate the performance of a Decision Tree model?**
    **Answer:** Evaluate performance using metrics such as accuracy, precision, recall, F1-score for classification tasks, and mean squared error or mean absolute error for regression tasks. Cross-validation provides a reliable estimate of performance.

32. **What is the ROC curve and how is it used in Decision Tree evaluation?**
    **Answer:** The ROC curve plots the true positive rate against the false positive rate at various threshold settings. It helps in evaluating the trade-off between sensitivity and specificity.

33. **What is the AUC-ROC metric and how is it interpreted?**
    **Answer:** The AUC-ROC (Area Under the ROC Curve) metric measures the overall performance of a binary classifier. Values close to 1 indicate excellent performance, while values close to 0.5 indicate poor performance.

34. **How do you interpret a confusion matrix for a Decision Tree model?**
    **Answer:** A confusion matrix shows the counts of true positives, true negatives, false positives, and false negatives. It helps in understanding the model's performance on each class.

35. **What is precision in the context of Decision Tree classification?**
    **Answer:** Precision is the ratio of true positives to the sum of true positives and false positives. It measures the accuracy of positive predictions.

36. **What is recall in the context of Decision Tree classification?**
    **Answer:** Recall (sensitivity) is the ratio of true positives to the sum of true positives and false negatives. It measures the ability of the model to identify positive cases.

37. **What is the F1-score and why is it important in Decision Tree evaluation?**
    **Answer:** The F1-score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall, especially useful for imbalanced datasets.

38. **How do you perform model selection in Decision Trees?**
    **Answer:** Model selection involves choosing the best hyperparameters based on cross-validation results and evaluation metrics. Techniques like grid search and randomized search can be used.

39. **What is the purpose of using a validation curve in Decision Tree evaluation?**
    **Answer:** Validation curves help in understanding how the model's performance varies with changes in hyperparameters. They are useful for identifying the optimal range of hyperparameters and diagnosing overfitting or underfitting.

40. **How do you interpret a high bias and high variance in Decision Trees?**
    **Answer:** High bias indicates underfitting, where the model is too simple to capture the underlying patterns. High variance indicates overfitting, where the model is too complex and captures noise in the data.

41. **What is pruning in the context of Decision Trees?**
    **Answer:** Pruning is the process of removing sections of the tree that provide little power in predicting target variables. It helps in reducing the complexity of the model and prevents overfitting.

42. **What are the different types of pruning techniques in Decision Trees?**
    **Answer:** Common pruning techniques include pre-pruning (stopping the tree from growing beyond a certain point) and post-pruning (removing branches from a fully grown tree).

43. **How do you perform post-pruning in Decision Trees?**
    **Answer:** Post-pruning involves growing the tree to its full depth and then removing nodes that provide the least information gain. This can be done by setting criteria such as a minimum error threshold.

44. **What is the purpose of a cost complexity pruning path in Decision Trees?**
    **Answer:** The cost complexity pruning path provides a sequence of trees obtained by pruning the original tree. It helps in selecting the optimal tree that balances model complexity and performance.

45. **How do you visualize the cost complexity pruning path in scikit-learn?**
    **Answer:** You can visualize the cost complexity pruning path using the `cost_complexity_pruning_path` function in scikit-learn. Example:
    ```python
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    ```

46. **What is the effect of using ensemble methods with Decision Trees?**
    **Answer:** Ensemble methods, such as Random Forests and Gradient Boosting, combine multiple Decision Trees to improve model performance, reduce variance, and increase robustness.

47. **How do Random Forests improve the performance of individual Decision Trees?**
    **Answer:** Random Forests improve performance by training multiple Decision Trees on different subsets of the data and features, and then averaging their predictions, reducing overfitting and increasing generalizability.

48. **What is the role of feature importance in evaluating Decision Trees?**
    **Answer:** Feature importance indicates the contribution of each feature to the model's predictions. It helps in understanding the model and identifying the most relevant features for the task.

49. **How do you handle class imbalance in Decision Tree classification?**
    **Answer:** Handle class imbalance by using techniques such as class weighting, oversampling the minority class, undersampling the majority class, or using ensemble methods designed to address imbalance.

50. **What is the impact of correlated features on Decision Trees?**
    **Answer:** Correlated features can lead to instability in the tree structure, as small changes in the data can result in different splits. Ensemble methods like Random Forests can help mitigate this issue.
