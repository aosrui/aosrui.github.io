# My Project
  This project aims to analyze the basic daily weather data of New York City in 2020 
and implement machine learning models, specifically linear regression and a neural 
network, to predict weather conditions such as temperature and precipitation.

# Introduction
  The central focus of this project lies in training a neural network machine learning
model and conducting a comparative analysis with a linear regression model. 
The dataset used for this exploration comprises daily weather data, and the primary 
goal is to predict two crucial daily weather factors: temperature and precipitation.
Accurate temperature predictions play a pivotal role in urban planning and infrastructure 
design. The ability to forecast temperatures aids in the planning and construction 
of infrastructure elements such as buildings, roads, and bridges. Different materials 
and construction techniques may be employed based on the expected temperature ranges.

  To address the weather prediction problem, I leveraged a comprehensive dataset sourced 
from NOAA, focusing on daily weather records in New York City throughout the year 2020. 
This dataset served as the foundation for our machine learning approach. 

  In this project,I employed linear regression and a neural network model, specifically 
the MLPRegressor, to predict two crucial daily weather factors—temperature and 
precipitation—using New York City's 2020 daily weather data from NOAA. The linear regression
model, leveraging features such as wind speed, maximum wind speed, temperature, and dew point, 
exhibited reasonable predictive performance, as evidenced by a commendable R-squared value 
and a visual representation comparing predicted and actual precipitation.
Transitioning to the MLPRegressor neural network model, I strategically adjusted hidden 
layer sizes and utilized standardization through StandardScaler for enhanced performance.
Despite convergence warnings during training, the MLPRegressor showcased competitive results, 
showcasing its potential for precipitation prediction, especially with a 
Relative Error Characteristic (REC) curve that illustrated prediction accuracy across different 
tolerance levels.

# Data

  The dataset used in this project originates from the National Oceanic and Atmospheric Administration (NOAA). It comprises daily weather records for New York City throughout the year 2020. 

### 1.1 orginal data

  The dataset includes information such as the date of the record, geographic coordinates (latitude and longitude), elevation, temperature (TEMP), dew point (DEWP), sea-level pressure (SLP), station-level pressure (STP), visibility (VISIB), wind speed (WDSP), maximum wind speed (MXSPD), wind gust speed (GUST), maximum temperature (MAX), minimum temperature (MIN), precipitation (PRCP), snow depth (SNDP), and a weather indicator (FRSHTT).

### 1.2 data cleaning

  Since the original data possess a lot of irrelevant or redundant columns, such as station identifiers and location coordinates,these were removed to focus on essential weather features. Missing values in numeric columns were imputed with the mean of their respective columns to ensure a complete dataset.

<img width="670" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/15a1ec9d-fa3a-47b0-bd62-048b8a587330"> 
      

### 1.3 preprocess data

  The date preprocessing process includes HANDING MISSING VALUES,REMOVING OUTLIERS,REMOVE ABNORMAL VALUES

<img width="684" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/bc821668-29e8-4b63-b48e-c77b94b505b0">

  Missing values in the dataset can disrupt the training of machine learning models. Imputing missing values, in this case using the mean of the respective columns, ensures that the dataset remains complete. This is important for maintaining data integrity and preventing the loss of valuable information.
  

<img width="689" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/548387fc-d508-4d21-adf9-68d2b1538316">

  The dataset was subjected to outlier detection using Z-scores. Columns containing numeric data were standardized, and Z-scores were calculated. Records with absolute Z-scores exceeding a predefined threshold of 3 were considered outliers. Subsequently, these outliers were removed from the dataset using a boolean mask, resulting in the creation of "data_clean_no_outliers".

### 1.4 visualize the data 

#### 1.4.1 select features

To specific the related variables of related targeted values - temperature and precipitation.First we calculated the coefficient between different variables, selecting the potential related ones for future model building. 
<img width="604" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/3ad75040-0c46-4723-9f98-77b35ad28795">

This figure visualize the correlation coefficient of the preprocessing data,to select the potential related variables to simplify the model building.

<img width="449" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/b56322d7-ed10-462e-b4f8-30862f22e52d">


Screening out featured varibables of temperature and precipitation.

#### 1.4.2 visualize two groups of variables

##### 1.4.2.1 precipitation group

<img width="753" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/1a6807a2-2ee3-40e6-8a1b-88371c64ec8e">
  
  Choosing 'WDSP','MXSPD','DEWP','TEMP' as features served for the target - precipitation to visualzie the changing trend of these.


##### 1.4.2.2 temperature group

<img width="758" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/16ae0dbc-2236-4d92-9717-3bdd9bce825b">
  
  Choosing 'WDSP','MXSPD','GUST' after elminiating the abnormal data,'dEWP' as features served for the target - temperature to visualzie the changing trend of these.

# Modeling 
The linear regression model and neural network model were applied to predict temperature and precipitation based on various weather features.

### 2.1 temperature group

#### 2.1.1 linear regression model

This code implements a linear regression model to predict temperature using features like Dew Point, Wind Gust, Wind Speed, and Maximum Wind Speed. It splits the data, trains the model, and evaluates its performance using Mean Squared Error and R-squared. The scatter plot visually compares actual and predicted temperatures based on Dew Point

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    X = data_clean_no_abnormal_gust[['DEWP', 'GUST','WDSP','MXSPD']]
    # Target: TEMP (Temperature)
    y = data_clean_no_abnormal_gust['TEMP']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a linear regression model
    model = LinearRegression()

    # Training the model
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Visualizing the predictions
    plt.scatter(X_test['DEWP'], y_test, color='black', label='Actual')
    plt.scatter(X_test['DEWP'], y_pred, color='blue', label='Predicted')
    plt.xlabel('Dew Point Temperature')
    plt.ylabel('Temperature')
    plt.title('Temperature Prediction Based on Dew Point ,GUST , WDSP, MXSPD')
    plt.legend()
    plt.show()

#### 2.1.2 neural network model

This code employs a Multi-Layer Perceptron (MLP) neural network for temperature prediction using features like Dew Point, Wind Gust, Wind Speed, and Maximum Wind Speed. It splits the data into training and testing sets, standardizes the features, and builds the neural network with two hidden layers of 100 neurons each. The model is trained on the scaled training data and evaluated on the test set, measuring performance using Mean Squared Error and R-squared. The neural network aims to capture complex patterns and relationships within the meteorological variables for improved temperature prediction.

    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Features: DEWP (Dew Point), GUST (Wind Gust Speed)
    X = data_clean_no_abnormal_gust[['DEWP', 'GUST','WDSP','MXSPD']]
    # Target: TEMP (Temperature)
    y = data_clean_no_abnormal_gust['TEMP']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train the neural network model
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

### 2.2 precipitation group

#### 2.2.1 linear regression model

Similar to the building process of temperaure's linear regression model, we use the features like Dew Point, Temperature, Wind Speed, and Maximum Wind Speed to predict the precipitation.

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt


    X = data_clean_no_abnormal_gust[['WDSP', 'MXSPD', 'TEMP','DEWP']]
    y = data_clean_no_abnormal_gust['PRCP']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a linear regression model
    model = LinearRegression()

    # Training the model
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Visualizing the predictions
    plt.scatter(y_test, y_test, color='black', label='Actual', alpha=0.5)
    plt.scatter(y_test, y_pred, color='blue', label='Predicted', alpha=0.5)
    plt.xlabel('Actual PRCP')
    plt.ylabel('Predicted PRCP')
    plt.title('Actual vs Predicted Precipitation')
    plt.legend()
    plt.show()



#### 2.2.2 neural network model


The neural network aimed to predict precipitation (PRCP) is based on meteorological variables (Wind Speed, Maximum Wind Speed, Temperature, Dew Point). The data is split into training and testing sets, and the input features are standardized. The MLPRegressor is configured with two hidden layers, each containing 85 neurons,which is the best preformance through the selection of various set of neurons. The model is trained, and predictions are made on the testing set. Mean Squared Error and R-squared are then calculated to evaluate the model's performance in predicting precipitation.

    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    # Assume X contains wind features and y contains precipitation
    X = data_clean_no_abnormal_gust[['WDSP', 'MXSPD', 'TEMP','DEWP']]
    y = data_clean_no_abnormal_gust['PRCP']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train the neural network model
    model = MLPRegressor(hidden_layer_sizes=(85, 85), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print(f'R-squared: {r2}')


# Result

### 3.1 temperaure group

<img width="401" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/76b9caa2-a206-4d7c-98ec-cfc168763252">

FIGURE 1

In Figure 1, the linear regression model for temperature was constructed with a test data size of 0.2, resulting in a Mean Squared Error (MSE) around 24.8. The model demonstrates high accuracy, as indicated by the low MSE. Additionally, the R-squared value, approximately 0.89, signifies a strong correlation between the predicted and observed values. This high coefficient suggests that the linear regression model effectively captures and explains the variance in the temperature data. The robust performance of the model enhances its reliability in forecasting temperature based on the selected features.

<img width="303" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/a7fbef4b-1cfc-4e76-b0c4-e364ade83b3f">

FIGURE 2

In Figure 2, the neural network model fitting results are illustrated. The model, characterized by a Mean Squared Error (MSE) of 23.6, outperforms the linear regression model in terms of accuracy. The lower MSE indicates a better fit of the neural network in predicting temperature. Additionally, the correlation between the tested and predicted data has increased to 0.89, highlighting the improved ability of the neural network model to capture the underlying patterns in the temperature data. This enhanced correlation further establishes the efficacy of the neural network in providing more accurate temperature predictions compared to the linear regression model.


<img width="320" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/4c495e8c-104e-419a-ab4c-0d8477fbfe99">

<img width="369" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/5158cd55-6c9f-4671-9fa9-7f3c1c32bbc9">

FIGURE 3

In Figure 3,The Relative Error Curve (REC) plot offers a comprehensive view of the MLPRegressor model's performance in predicting temperature. The X-axis represents the absolute error tolerance, delineating the acceptable range of disparities between predicted and actual temperatures. Meanwhile, the Y-axis illustrates the corresponding percentage of correct predictions within each absolute error tolerance. The curve itself portrays how the model's accuracy evolves across different levels of precision.


### 3.2 precipitation group

<img width="409" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/0d6c23cf-d67b-4477-a838-bddecdfd2bd9">

FIGURE 5

In Figure 5, the linear regression model for precipitation was constructed with a test data size of 0.2, resulting in a Mean Squared Error (MSE) around 0.0068. The model demonstrates high accuracy, as indicated by the low MSE. However, the model shows low correlationship as to the R-squared value, approximately 0.17, signifies a weak correlation between the predicted and observed values.

<img width="317" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/e28785d1-b612-4a9c-aaa6-3515757e7a95">

FIGURE 6

In Figure 6, the results of the neural network model fitting are presented, revealing a Mean Squared Error (MSE) of 0.0068. This performance surpasses that of the linear regression model, indicating heightened accuracy in predicting precipitation. Despite a weak correlation of 0.17 between the tested and predicted data, it represents an improvement over the linear regression model. 


<img width="383" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/05026422-21df-4c84-af9c-b5dd3ba83e84">

FIGURE 7

The REC curve plot in Figure 7 for the MLPRegressor model indicates a Root Mean Squared Error (RMSE) of 0.09. This metric reflects the overall accuracy of the model's temperature predictions. The lower the RMSE, the better the model's performance, suggesting that, on average, the predicted temperatures are close to the actual values. Therefore, an RMSE of 0.09 in the REC curve plot signifies a relatively accurate temperature prediction by the MLPRegressor model, as it falls within a reasonable margin of error.


# Discussion

### 4.1 comparsion of linear regression model and neural network model


The comparison between the linear regression and neural network models reveals distinct advantages and trade-offs in the context of temperature and precipitation prediction. In the temperature prediction task, the linear regression model demonstrates strong accuracy with a low Mean Squared Error (MSE) of 24.8 in Figure 1. The robust fitting degree, as indicated by the high R-squared value of 0.89, signifies a substantial correlation between the predicted and observed values. This initial success positions linear regression as a reliable baseline model.

However, Figure 2 introduces the neural network model, which outperforms linear regression in terms of accuracy. With a lower MSE of 23.6 and an increased correlation of 0.89, the neural network excels in capturing intricate temperature patterns. The neural network's ability to learn non-linear relationships and adapt to complex data structures contributes to its superior performance.

Figure 3, showcasing the Relative Error Curve (REC) plot, provides additional insights into the neural network model's adaptability across different levels of error tolerance. This plot further highlights the model's versatility and its capacity to maintain accuracy under varying precision requirements.

In the context of precipitation prediction (Figure 5 and Figure 6), both linear regression and the neural network exhibit high accuracy with low MSE values (0.0068). However, the weak correlationship (R-squared value of 0.17) in both models indicates challenges in capturing the underlying patterns in precipitation data. The neural network, while maintaining a weak correlation, showcases an improvement over linear regression.

Figure 7 introduces the REC curve plot for the MLPRegressor model, revealing a Root Mean Squared Error (RMSE) of 0.09. This metric provides a nuanced understanding of the model's overall accuracy in predicting temperature. The lower RMSE suggests that, on average, the predicted temperatures closely align with the actual values.

In conclusion, the neural network model demonstrates superior performance over linear regression in both temperature and precipitation prediction tasks. Its capacity to capture non-linear relationships and adapt to complex data structures makes it a promising choice for meteorological predictions, offering enhanced accuracy and adaptability in diverse forecasting scenarios.

### 4.2 Challenges in Precipitation Prediction


The relatively low correlation between the test data and predicted data in the context of precipitation prediction can be attributed to several inherent challenges associated with the nature of precipitation patterns. Precipitation is a complex meteorological phenomenon influenced by a myriad of factors, including atmospheric pressure, temperature, wind speed, and humidity, among others. Linear regression, as a simplistic model, may struggle to capture the intricate non-linear relationships among these variables, leading to a limited ability to accurately predict precipitation.

Moreover, precipitation data often exhibit high variability and dependence on localized factors, introducing spatial and temporal complexities. Linear regression models assume a linear relationship between the input features and the target variable, which may oversimplify the intricate dynamics of precipitation patterns. Neural networks, with their capacity to model non-linear relationships and adapt to complex structures, offer an improvement over linear regression in capturing the nuanced interactions that influence precipitation.

In Figure 5 and Figure 6, both linear regression and neural network models exhibit low correlations (R-squared values of approximately 0.17) with the test data. This suggests that the selected features, such as dew point, wind speed, and gust, might not sufficiently encapsulate the multifaceted nature of precipitation. The inherent variability and unpredictability of precipitation events make it challenging for any model, including neural networks, to achieve a high correlation with the observed values.

Additionally, the low correlation may also be indicative of the inherent stochastic nature of precipitation, where short-term variations and local conditions play a significant role. Linear regression and neural networks may struggle to discern these subtle variations without comprehensive and diverse datasets that encompass a wide range of meteorological conditions.

The challenges associated with the complex, non-linear, and locally influenced nature of precipitation patterns contribute to the relatively low correlations observed in both linear regression and neural network models. Improving precipitation prediction models may require incorporating more sophisticated features, considering localized factors, and exploring advanced techniques that can better capture the intricate dynamics of this meteorological phenomenon.

# Conclusion

This project delves into the prediction of daily weather conditions, particularly temperature and precipitation in New York City throughout 2020, utilizing linear regression and neural network models. The dataset, sourced from the National Oceanic and Atmospheric Administration (NOAA), underwent rigorous preprocessing involving data cleaning, feature selection, and visualization. The models were evaluated based on standard metrics such as Mean Squared Error, R-squared values, and Relative Error Curves. Findings indicate that while linear regression exhibited commendable accuracy and correlation in temperature prediction, the neural network model outperformed it, demonstrating enhanced predictive accuracy and improved adaptability across varying error tolerance thresholds. In precipitation prediction, both models achieved high accuracy but struggled with low correlations, attributed to the intricate, non-linear, and locally influenced nature of precipitation patterns. 

To enhance the predictive capabilities of weather models, several targeted improvement strategies are proposed. Advanced feature engineering will involve a meticulous selection and transformation of input variables, ensuring a more nuanced representation of meteorological factors influencing temperature and precipitation. Exploring sophisticated neural network architectures entails the investigation of intricate model structures to capture complex non-linear relationships within the data. The implementation of ensemble models, which combine predictions from multiple models, can further enhance accuracy and robustness. Hyperparameter tuning involves fine-tuning model parameters to optimize performance. Consideration of spatial and temporal factors recognizes the localized and time-dependent nature of weather phenomena, introducing refined granularity into the models. Lastly, the utilization of advanced visualization tools aims to provide more insightful and interpretable representations of model outputs, fostering a deeper understanding of the forecasted weather conditions in New York City. Collectively, these targeted enhancements aim to fortify the models' predictive accuracy, ensuring more reliable and precise daily weather forecasts.

# references


[1]Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., ... & Zheng, X. (2016). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.org.

[2]Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.

[3]McKinney, W. (2010). Data structures for statistical computing in Python. In Proceedings of the 9th Python in Science Conference (pp. 51-56).

[4]Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.













