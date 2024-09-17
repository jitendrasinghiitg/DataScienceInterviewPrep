# Classification Model Evaluation Metrics

1. What is a confusion matrix?
   - A table that describes the performance of a classification model
   - Shows the counts of true positives, true negatives, false positives, and false negatives
   - Basis for many classification evaluation metrics

2. What are true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)?
   - TP: Correctly predicted positive instances
   - TN: Correctly predicted negative instances
   - FP: Negative instances incorrectly predicted as positive
   - FN: Positive instances incorrectly predicted as negative

3. What is accuracy and how is it calculated?
   - The proportion of correct predictions (both positive and negative) among the total number of cases examined
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Range: 0 to 1, with 1 being perfect accuracy

4. What is the limitation of accuracy as a metric?
   - It can be misleading for imbalanced datasets
   - Doesn't distinguish between types of errors (FP vs FN)

5. What is precision and how is it calculated?
   - The proportion of correct positive predictions among all positive predictions
   - Formula: TP / (TP + FP)
   - Also known as Positive Predictive Value (PPV)

6. What is recall and how is it calculated?
   - The proportion of correct positive predictions among all actual positive instances
   - Formula: TP / (TP + FN)
   - Also known as sensitivity or True Positive Rate (TPR)

7. What is the F1 score?
   - The harmonic mean of precision and recall
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)
   - Provides a single score that balances both precision and recall

8. When would you use F1 score instead of accuracy?
   - When you have an uneven class distribution (imbalanced dataset)
   - When you want to find an optimal balance between precision and recall

9. What is the difference between macro-average and micro-average in multi-class classification?
   - Macro-average: Compute the metric independently for each class and then take the average
   - Micro-average: Aggregate the contributions of all classes to compute the average metric

10. What is specificity and how is it calculated?
    - The proportion of actual negative instances that were correctly identified
    - Formula: TN / (TN + FP)
    - Also known as True Negative Rate (TNR)

11. What is the Receiver Operating Characteristic (ROC) curve?
    - A plot of True Positive Rate (recall) against False Positive Rate at various threshold settings
    - Used to show the tradeoff between sensitivity and specificity
    - The area under the ROC curve (AUC-ROC) is used as a summary statistic

12. What does the Area Under the ROC Curve (AUC-ROC) represent?
    - A measure of the model's ability to distinguish between classes
    - Range: 0.5 to 1, where 0.5 is random guessing and 1 is perfect classification
    - Insensitive to class imbalance, unlike accuracy

13. What is the Precision-Recall (PR) curve?
    - A plot of precision against recall at various threshold settings
    - Particularly useful for imbalanced datasets
    - The area under the PR curve (AUC-PR) is used as a summary statistic

14. When would you prefer the PR curve over the ROC curve?
    - When dealing with highly imbalanced datasets
    - When you're more interested in positive class performance

15. What is log loss (logarithmic loss) and how is it used?
    - A classification loss function that quantifies the accuracy of predictions based on how confident the model is
    - Heavily penalizes confident and wrong predictions
    - Lower values indicate better model performance

16. What is the Brier score?
    - A measure of the mean squared difference between the predicted probability and the actual outcome
    - Range: 0 to 1, with 0 being perfect
    - Used for evaluating probabilistic predictions

17. How does the Gini coefficient relate to the AUC-ROC?
    - Gini coefficient = 2 * AUC-ROC - 1
    - Range: 0 to 1, with 1 being perfect separation
    - Often used in credit scoring models

18. What is Cohen's Kappa and when is it useful?
    - A statistic that measures inter-rater agreement for categorical items
    - Takes into account the possibility of agreement occurring by chance
    - Useful when you want to know how much better your model is performing over random chance

19. What is the Matthews Correlation Coefficient (MCC)?
    - A measure of the quality of binary classifications
    - Takes into account true and false positives and negatives
    - Considered a balanced measure even if classes are of very different sizes

20. How do you handle multi-class classification metrics?
    - Use micro, macro, or weighted averaging of binary metrics
    - Use specific multi-class metrics like multi-class log loss
    - Consider one-vs-rest or one-vs-one approaches for some metrics

21. What is cross-entropy loss and how is it used in classification?
    - A loss function that measures the performance of a classification model whose output is a probability value between 0 and 1
    - Increases as the predicted probability diverges from the actual label
    - Commonly used as the loss function for training neural networks for classification

22. What is the difference between hard and soft classification metrics?
    - Hard classification metrics work with class labels (0 or 1)
    - Soft classification metrics work with probabilities or scores
    - Soft metrics (like AUC-ROC, log loss) often provide more information about model confidence

23. What is calibration in the context of classification models?
    - The alignment of predicted probabilities with observed frequencies
    - A well-calibrated model's predicted probability of 0.8 should be correct 80% of the time
    - Can be visualized using reliability diagrams or assessed using the Brier score

24. What is the Kolmogorov-Smirnov (K-S) statistic in classification?
    - Measures the maximum difference between the cumulative distribution functions of the positive and negative classes
    - Range: 0 to 1, with higher values indicating better separation between classes
    - Often used in credit scoring and financial applications

25. What is lift and how is it used in classification evaluation?
    - Measures how much better a model performs compared to a random model
    - Often used in marketing to evaluate the effectiveness of predictive models
    - Calculated as the ratio of the results obtained with and without the model

26. What is the Jaccard index (Intersection over Union) and when is it used?
    - Measures the overlap between predicted and actual positive instances
    - Formula: TP / (TP + FP + FN)
    - Often used in image segmentation and document classification

27. How do you evaluate multi-label classification models?
    - Use label-based metrics (e.g., hamming loss)
    - Use example-based metrics (e.g., subset accuracy)
    - Consider label correlations and hierarchies if applicable

28. What is the difference between micro-F1 and macro-F1 scores?
    - Micro-F1: Calculates F1 using global TP, FP, and FN counts
    - Macro-F1: Calculates F1 for each class and then takes the average
    - Micro-F1 gives equal weight to each instance, macro-F1 gives equal weight to each class

29. How do you handle class imbalance when evaluating classification models?
    - Use metrics less sensitive to imbalance (e.g., F1-score, AUC-ROC)
    - Consider precision-recall curve instead of ROC curve
    - Use stratified sampling in cross-validation
    - Apply techniques like oversampling, undersampling, or SMOTE before evaluation

30. What is the importance of threshold selection in binary classification?
    - Different thresholds can significantly affect precision, recall, and overall performance
    - Optimal threshold depends on the specific problem and costs of different types of errors
    - Techniques like ROC analysis and precision-recall curves can help in threshold selection