This project is planned by @plan.md, and the results are shown in
@run_pipeline.ipynb. I am not satisfied with this structure and the outcome.

I now want to completely redo this project. This time I want the following:

1. Instead of using the LightGBM features to execute a long short trading book or
   whatever strategy, just measure the predictive power of our ensemble learning
   model as the final result. Because the preditive power of our model can be
   directly transform into trading book.

2. In stead of targeting 50 securities, now only target on the 台灣加權指數 (TWII), that's it, the benchmark is also itself. That is, we are now training a ensemble learning model to predict the future price of TWII, and the performance of our model is measured by the predictive power, which is directly related to the trading book performance. Therefore no need to construct a trading book. Only focus on the predictive power of our model.

3. All the candidate features should be the technical price indicators, which are listed here @technical-indicator.md. I will not use any other features, such as sentiment analysis or macroeconomic indicators to make things simpler.

4. This project will utilize time series forecasting for stock market index prediction, where the models will predict output y at time t+1 where the lookback window is k = 60 days. That is, y\_(t+1) = f(y\_(t-k:y), x\_(t-k:t)), where x is the technical indicators. The performance of the model will be evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to assess the accuracy of the predictions.

5. Feature Scaling: Quantile-based Scaling: Scaling is applied through standardization for all features. While this is not needed in practice for GBMs since they are decision trees which use gradient bosting and not gradient descent, scaling is still applied since RNNs like GRU and LSTM require it to achieve faster convergence rates. To deal with outliers in the data, a quantile-based scaler is used to scale all features with the 25th quantile as 0 and the 75th quantile as 1 so that the scaling is not disproportionately influenced by very large outliers.

6. The training set should be 85% of the data, the validation set should be 5% of the data, and the test set should be 10% of the data. The training set will be used to train the model, the validation set will be used to tune the hyperparameters and select the best model, and the test set will be used to evaluate the final performance of the model.

7. If the legacy code is not reusable, I will just throw it away and start from scratch. I will not try to reuse the legacy code if it is not reusable, because that will just make things more complicated and messy. I want to have a clean and simple codebase that is easy to understand and maintain.
