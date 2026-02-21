# ml-agriculture-insights
An investigation into the various environmental and management factors influencing crop yield.

## Goal 
The goal of the project is to architect a system that is able to assess how various environmental as well as management factors can influence crop production across the different geographic regions. Because of the increasing global demand for food security on top of the impact that climate change can have on agriculture, there is a relationship in these areas, and it is important to understand exactly what and to visualize these connections. Hence, we can use machine learning and parallel computing techniques to build a system that is able to identify which of the environmental inputs like rainfall, temperature, humidity, etc., as well as the management practices (pesticide usage, etc.) as per the dataset, would most significantly affect the crop yield outcomes in the different locations and over time.


## Dataset 
The dataset that was chosen consists of agricultural records that specifically have key environmental management, as well as production-related factors across the various geographic regions across the years. It makes it easy to carry out analysis of how the external variables can affect the crop yields over time. 

The data is time-series and multiregional, meaning that it covers multiple years and countries and hence makes it suitable for a temporal and spatial trend analysis. Additionally, there are multiple independent variables and one dependent variable. The data is also structured and ready to use in regression models. 

### Justification of the Dataset

Hence, the selected dataset is extremely relevant and very suited for the objectives of the project - this is because it aims to assess what impact environmental and management factors can have on crop production across the multiple geographic regions highlighted in the dataset. 

#### Environmental and Management Data - 
Crop yield data is provided, which is the main outcome that we want to predict and analyze for any relationships. 
The environmental variables, like average temperature and rainfall, are also provided, and these factors are well known to impact agricultural productivity. 
There is also management input such as pesticide usage, and this highlights the impact humans can have in technological advancements. 
These factors allow us to evaluate how both natural conditions and farming strategies can impact crop production, and the core requirement of assessing environmental and management factors is effectively satisfied. 

#### Geographical and Temporal Data
The dataset spans multiple countries (‘Area’), enabling cross-regional comparisons of the crop performances and the farming practices. 
Furthermore, it spans multiple years (Year), and the time-series data would allow us to assess what trends there are, and the impacts of climate change on production, and highlight how effective certain strategies are over time. 
These factors contribute to making the dataset ideal to build a generalizable model where it can work across multiple regions and can be effectively used to predict future outcomes. 

#### Machine Learning Support and Parallel Processing
From a technical standpoint: 
The dataset is structured and cleaned, which allows it to be used in machine learning models and very efficiently so. 
Numerical and categorical features are included which means that it supports diverse preprocessing and encoding techniques. 

#### Data Volume and Quality 
The data contains over 28,000 observations, which hence provides the sufficient data for training robust machine learning models and hence reduces the risk of overfitting. 
The initial data exploration shows that there is no missing value or duplicates which speeds up the process of training the models and streamlines the workflow.

#### Features Richness 
The dataset includes specific crop types (‘item’) in addition to the environmental and management features, and hence allows for more detailed insights on how different crops can respond to the varying conditions. 
Thus, the granularity would support the development of crop-specific strategies for yield improvement rather than a model that is more towards ‘one-size-fits-all’. 

#### Predictive and Analytical Value 
The dataset has a well-defined target variable (hg/ha_yield) and the relevant features which makes it ideal for building predictive models like Random Forest Regressors. 
These models would allow us to forecast the crop yields under the varying environmental and management conditions which is valuable for proactive agricultural planning. 

Combined, these points can conform very well into real-world applications. Specifically, the dataset using real-world agricultural metrics makes it very interpretable to domain experts (eg. yields in hg/ha, pesticide use in tonnes), and makes it very useful for decision-making in agricultural policy, resource planning, or research in general. 

