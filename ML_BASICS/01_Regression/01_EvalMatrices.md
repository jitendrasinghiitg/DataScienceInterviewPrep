# Regression Model Evaluation Metrics:

1. What is the purpose of regression model evaluation metrics?
   - To assess how well a regression model fits the data and predicts new observations
   - To compare different regression models and select the best one
   - To identify areas where the model may be improved

2. What is Mean Squared Error (MSE)?
   - The average of the squared differences between predicted and actual values
   - Formula: MSE = (1/n) * Σ(y_i - ŷ_i)^2, where y_i is the actual value and ŷ_i is the predicted value
   - Lower MSE indicates better model performance

3. What is Root Mean Squared Error (RMSE)?
   - The square root of the Mean Squared Error
   - Formula: RMSE = √MSE
   - Expressed in the same units as the dependent variable, making it more interpretable than MSE

4. What is Mean Absolute Error (MAE)?
   - The average of the absolute differences between predicted and actual values
   - Formula: MAE = (1/n) * Σ|y_i - ŷ_i|
   - Less sensitive to outliers compared to MSE/RMSE

5. When would you prefer MAE over RMSE?
   - When you want to treat all errors equally, regardless of their magnitude
   - When working with datasets that contain outliers
   - When you need a metric that's more robust to outliers

6. What is R-squared (R²) and what does it represent?
   - Also known as the coefficient of determination
   - Represents the proportion of variance in the dependent variable explained by the model
   - Ranges from 0 to 1, with 1 indicating perfect fit

7. How is R-squared calculated?
   - R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)
   - Alternatively: R² = 1 - (Unexplained Variation / Total Variation)

8. What are the limitations of R-squared?
   - It always increases when new predictors are added, even if they're not meaningful
   - It doesn't indicate whether the coefficients are statistically significant
   - It doesn't indicate whether the model is biased

9. What is Adjusted R-squared?
   - A modified version of R-squared that adjusts for the number of predictors in the model
   - Penalizes the addition of unnecessary predictors
   - Formula: Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)], where n is sample size and k is number of predictors

10. When would you use Adjusted R-squared instead of R-squared?
    - When comparing models with different numbers of predictors
    - To assess whether adding a new predictor improves the model more than would be expected by chance

11. What is Mean Absolute Percentage Error (MAPE)?
    - The average of the absolute percentage differences between predicted and actual values
    - Formula: MAPE = (100/n) * Σ|(y_i - ŷ_i) / y_i|
    - Expressed as a percentage, making it easy to interpret

12. What is a limitation of MAPE?
    - It can't be used when actual values are zero or close to zero
    - It puts a heavier penalty on negative errors than on positive errors

13. What is Symmetric Mean Absolute Percentage Error (SMAPE)?
    - A variation of MAPE that addresses some of its limitations
    - Formula: SMAPE = (100/n) * Σ(|y_i - ŷ_i| / ((|y_i| + |ŷ_i|) / 2))
    - Bounded between 0% and 200%

14. What is the Coefficient of Variation (CV)?
    - A standardized measure of dispersion of a probability distribution or frequency distribution
    - Formula: CV = (RMSE / mean of actual values) * 100
    - Useful for comparing models with different scales

15. What is Residual Standard Error (RSE)?
    - An estimate of the standard deviation of the residuals
    - Formula: RSE = √(Sum of Squared Residuals / (n - k - 1))
    - Used to calculate prediction intervals

16. What is the difference between in-sample and out-of-sample error?
    - In-sample error is calculated on the data used to train the model
    - Out-of-sample error is calculated on new, unseen data
    - Out-of-sample error is generally a better indicator of model performance

17. What is cross-validation and why is it used in model evaluation?
    - A resampling method that uses different portions of the data to test and train a model on different iterations
    - Provides a more robust estimate of model performance
    - Helps to detect overfitting

18. What is K-fold cross-validation?
    - A cross-validation method where the data is divided into K subsets
    - The model is trained on K-1 subsets and tested on the remaining subset
    - This process is repeated K times, with each subset serving as the test set once

19. What is the bias-variance tradeoff in the context of model evaluation?
    - Bias: The error from erroneous assumptions in the learning algorithm
    - Variance: The error from sensitivity to small fluctuations in the training set
    - A good model balances these two sources of error

20. How can you use the bias-variance tradeoff to diagnose overfitting or underfitting?
    - High bias and low variance suggests underfitting
    - Low bias and high variance suggests overfitting
    - The goal is to find the sweet spot with the right balance

21. What is the Akaike Information Criterion (AIC)?
    - A metric for comparing the quality of statistical models
    - Balances model fit against model complexity
    - Lower AIC indicates a better model

22. What is the Bayesian Information Criterion (BIC)?
    - Similar to AIC, but with a stronger penalty for model complexity
    - Generally prefers simpler models compared to AIC
    - Lower BIC indicates a better model

23. When would you use AIC vs BIC?
    - Use AIC when false negatives are more concerning (missing a true relationship)
    - Use BIC when false positives are more concerning (including a spurious relationship)

24. What is the Mallows' Cp statistic?
    - A metric used to assess the fit of a regression model
    - Helps to choose the best subset of predictors
    - A model with Cp close to p (number of predictors including intercept) is considered good

25. What is heteroscedasticity and how does it affect model evaluation?
    - Heteroscedasticity is when the variability of the residuals is not constant across all levels of the independent variables
    - It can lead to inefficient parameter estimates and incorrect standard errors
    - It may be detected through residual plots or statistical tests

26. What is multicollinearity and how does it impact model evaluation?
    - Multicollinearity is high correlation between independent variables
    - It can lead to unstable and unreliable coefficient estimates
    - It may be detected through correlation matrices or Variance Inflation Factor (VIF)

27. What is the purpose of a Q-Q plot in regression diagnostics?
    - To assess whether the residuals follow a normal distribution
    - Points should roughly follow a straight line for normally distributed residuals

28. What is Cook's distance and how is it used in regression diagnostics?
    - A measure of the influence of each observation on the regression results
    - Helps identify influential points that may be skewing the model
    - Points with high Cook's distance may warrant further investigation

29. What is the difference between parametric and non-parametric measures of model fit?
    - Parametric measures assume a specific distribution of the data (often normal)
    - Non-parametric measures make fewer assumptions about the data distribution
    - Non-parametric measures are often more robust but may have less statistical power

30. How can you use bootstrapping for model evaluation?
    - Bootstrapping involves resampling the data with replacement many times
    - It can be used to estimate the sampling distribution of model parameters
    - Provides a way to calculate confidence intervals for model metrics