# Logistic Regression: 30 Concept-Clearing Questions and Answers

1. What is logistic regression?
   - A statistical method for predicting a binary outcome based on one or more independent variables
   - Used for classification problems where the dependent variable is categorical

2. How does logistic regression differ from linear regression?
   - Logistic regression predicts probabilities of categorical outcomes
   - Linear regression predicts continuous numerical outcomes

3. What is the logistic function (sigmoid function)?
   - S-shaped curve that maps any real-valued number to a value between 0 and 1
   - Formula: f(x) = 1 / (1 + e^(-x))

4. Why is the sigmoid function used in logistic regression?
   - It transforms the linear combination of predictors into a probability (0 to 1 range)
   - Ensures output is interpretable as a probability for binary classification

5. What does the output of logistic regression represent?
   - The probability of belonging to the positive class (usually labeled as 1)

6. How is the decision boundary determined in logistic regression?
   - Typically set at 0.5 probability
   - Predictions above 0.5 are classified as positive, below as negative

7. What is the cost function used in logistic regression?
   - Log loss (binary cross-entropy)
   - Measures the difference between predicted probabilities and actual class labels

8. Why can't we use mean squared error (MSE) as the cost function for logistic regression?
   - MSE would result in a non-convex optimization problem
   - Log loss provides a convex optimization problem, easier to solve

9. What optimization algorithm is commonly used to fit logistic regression models?
   - Gradient descent or its variants (e.g., stochastic gradient descent)

10. What is maximum likelihood estimation in the context of logistic regression?
    - Method to estimate model parameters by maximizing the likelihood of observing the given data

11. What are the assumptions of logistic regression?
    - Independence of observations
    - Linearity in the log odds
    - No multicollinearity among predictors
    - Large sample size

12. How do you interpret the coefficients in logistic regression?
    - They represent the change in log odds of the outcome for a one-unit change in the predictor

13. What is odds ratio in logistic regression?
    - Exponential of the coefficient
    - Represents the change in odds of the outcome for a one-unit change in the predictor

14. How do you handle multiclass classification with logistic regression?
    - One-vs-Rest (OvR) approach
    - Multinomial logistic regression

15. What is the difference between L1 and L2 regularization in logistic regression?
    - L1 (Lasso) can lead to sparse models by driving some coefficients to zero
    - L2 (Ridge) shrinks all coefficients towards zero but doesn't eliminate them

16. What is the purpose of regularization in logistic regression?
    - Prevents overfitting by adding a penalty term to the cost function
    - Helps handle multicollinearity

17. How do you handle imbalanced datasets in logistic regression?
    - Adjust class weights
    - Oversampling minority class or undersampling majority class
    - Use techniques like SMOTE (Synthetic Minority Over-sampling Technique)

18. What is the ROC curve in the context of logistic regression?
    - Plots True Positive Rate against False Positive Rate at various threshold settings
    - Used to evaluate the model's ability to discriminate between classes

19. What does AUC-ROC represent?
    - Area Under the ROC Curve
    - Measure of the model's ability to distinguish between classes (0.5 = random, 1 = perfect)

20. How do you handle categorical variables in logistic regression?
    - One-hot encoding for nominal variables
    - Ordinal encoding for ordinal variables

21. What is multicollinearity and why is it a problem in logistic regression?
    - High correlation between predictor variables
    - Can lead to unstable and unreliable coefficient estimates

22. How can you detect multicollinearity?
    - Correlation matrix
    - Variance Inflation Factor (VIF)

23. What is the difference between parametric and non-parametric logistic regression?
    - Parametric assumes a specific functional form for the relationship between predictors and outcome
    - Non-parametric makes fewer assumptions about the underlying relationship

24. What is the link function in generalized linear models, and what is it for logistic regression?
    - Function that relates the linear predictor to the expected value of the outcome
    - For logistic regression, it's the logit function: log(p / (1-p))

25. How do you handle missing data in logistic regression?
    - Complete case analysis (dropping missing values)
    - Imputation techniques (mean, median, or more advanced methods)
    - Using algorithms that can handle missing values (e.g., some implementations of random forests)

26. What is the difference between discriminative and generative models, and where does logistic regression fit?
    - Discriminative models learn the decision boundary between classes
    - Generative models learn the distribution of each class
    - Logistic regression is a discriminative model

27. How do you assess the goodness of fit for a logistic regression model?
    - Deviance
    - Hosmer-Lemeshow test
    - Pseudo R-squared measures (e.g., McFadden's R-squared)

28. What is the difference between logistic regression and probit regression?
    - Logistic regression uses the logistic function as the link function
    - Probit regression uses the cumulative normal distribution as the link function
    - They often produce similar results but can differ in extreme cases

29. How do you handle interactions between variables in logistic regression?
    - Include interaction terms as additional predictors in the model
    - Interpret coefficients carefully as they now represent conditional effects

30. What are the advantages and disadvantages of logistic regression?
    Advantages:
    - Simple and interpretable
    - Efficient to train
    - Performs well on linearly separable classes
    Disadvantages:
    - Assumes linearity in log odds
    - May underperform with complex non-linear relationships
    - Sensitive to outliers and multicollinearity