# My Project
  This project aims to analyze the basic daily weather data of New York City in 2020 
and implement machine learning models, specifically linear regression and a neural 
network, to predict weather conditions such as temperature and precipitation.

# Introuction
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

1.1 orginal data

  The dataset includes information such as the date of the record, geographic coordinates (latitude and longitude), elevation, temperature (TEMP), dew point (DEWP), sea-level pressure (SLP), station-level pressure (STP), visibility (VISIB), wind speed (WDSP), maximum wind speed (MXSPD), wind gust speed (GUST), maximum temperature (MAX), minimum temperature (MIN), precipitation (PRCP), snow depth (SNDP), and a weather indicator (FRSHTT).

1.2 data cleaning

  Since the original data possess a lot of irrelevant or redundant columns, such as station identifiers and location coordinates,these were removed to focus on essential weather features. Missing values in numeric columns were imputed with the mean of their respective columns to ensure a complete dataset.

<img width="670" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/15a1ec9d-fa3a-47b0-bd62-048b8a587330"> 
      

1.3 preprocess data

  The date preprocessing process includes HANDING MISSING VALUES,REMOVING OUTLIERS,REMOVE ABNORMAL VALUES

<img width="684" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/bc821668-29e8-4b63-b48e-c77b94b505b0">
  Missing values in the dataset can disrupt the training of machine learning models. Imputing missing values, in this case using the mean of the respective columns, ensures that the dataset remains complete. This is important for maintaining data integrity and preventing the loss of valuable information.

<img width="689" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/548387fc-d508-4d21-adf9-68d2b1538316">
  The dataset was subjected to outlier detection using Z-scores. Columns containing numeric data were standardized, and Z-scores were calculated. Records with absolute Z-scores exceeding a predefined threshold of 3 were considered outliers. Subsequently, these outliers were removed from the dataset using a boolean mask, resulting in the creation of "data_clean_no_outliers".

1.4 visualize the data 

1.4.1 select features

To specific the related variables of related targeted values - temperature and precipitation.First we calculated the coefficient between different variables, selecting the potential related ones for future model building. 
<img width="604" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/3ad75040-0c46-4723-9f98-77b35ad28795">

This figure visualize the correlation coefficient of the preprocessing data,to select the potential related variables to simplify the model building.

<img width="449" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/b56322d7-ed10-462e-b4f8-30862f22e52d">

Screening out featured varibables of temperature and precipitation.

1.4.2 visualize two groups of variables

1.4.2.1 precipitation group

<img width="753" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/1a6807a2-2ee3-40e6-8a1b-88371c64ec8e">
  Choosing 'WDSP','MXSPD','DEWP','TEMP' as features served for the target - precipitation to visualzie the changing trend of these.


1.4.2.2 temperature group

<img width="758" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/16ae0dbc-2236-4d92-9717-3bdd9bce825b">
  Choosing 'WDSP','MXSPD','GUST' after elminiating the abnormal data,'dEWP' as features served for the target - temperature to visualzie the changing trend of these.

# Modeling 
The linear regression model and neural network model were applied to predict temperature and precipitation based on various weather features.

2.1 temperature group

2.1.1 linear regression model

This code implements a linear regression model to predict temperature using features like Dew Point, Wind Gust, Wind Speed, and Maximum Wind Speed. It splits the data, trains the model, and evaluates its performance using Mean Squared Error and R-squared. The scatter plot visually compares actual and predicted temperatures based on Dew Point
<img width="761" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/317addd9-5de1-47ab-aa57-a563793112ba">

2.1.2 neural network model

This code employs a Multi-Layer Perceptron (MLP) neural network for temperature prediction using features like Dew Point, Wind Gust, Wind Speed, and Maximum Wind Speed. It splits the data into training and testing sets, standardizes the features, and builds the neural network with two hidden layers of 100 neurons each. The model is trained on the scaled training data and evaluated on the test set, measuring performance using Mean Squared Error and R-squared. The neural network aims to capture complex patterns and relationships within the meteorological variables for improved temperature prediction.
<img width="696" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/fe1ee1e4-4064-4373-bd29-5e56ccbf725e">

2.2 precipitation group

2.2.1 linear regression model

Similar to the building process of temperaure's linear regression model, we use the features like Dew Point, Temperature, Wind Speed, and Maximum Wind Speed to predict the precipitation.

<img width="683" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/1610b0a0-58b2-41f0-9b6c-90079d62e459">


2.2.2 neural network model


The neural network aimed to predict precipitation (PRCP) is based on meteorological variables (Wind Speed, Maximum Wind Speed, Temperature, Dew Point). The data is split into training and testing sets, and the input features are standardized. The MLPRegressor is configured with two hidden layers, each containing 85 neurons. The model is trained, and predictions are made on the testing set. Mean Squared Error and R-squared are then calculated to evaluate the model's performance in predicting precipitation.

<img width="689" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/5ac1a480-4baa-47e7-8aa6-cba5079ebd79">

# Result

3.1 temperaure group

<img width="401" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/76b9caa2-a206-4d7c-98ec-cfc168763252">

<img width="303" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/a7fbef4b-1cfc-4e76-b0c4-e364ade83b3f">

<img width="325" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/4b97f566-c916-4b27-a946-edd5624a3bfb">

<img width="337" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/b88d26c2-ba1f-46e0-a9bc-ffdda38f9076">


3.2 precipitation group

<img width="409" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/0d6c23cf-d67b-4477-a838-bddecdfd2bd9">

<img width="317" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/e28785d1-b612-4a9c-aaa6-3515757e7a95">

<img width="383" alt="image" src="https://github.com/aosrui/aosrui.github.io/assets/152749873/05026422-21df-4c84-af9c-b5dd3ba83e84">


# Discussion

# Conclusion

# references













