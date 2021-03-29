# Capstone 2

Speak to Hamid
Possible concerns from Michelle:
If time series dependent, might be better for capstone 3
Might be best to model just end of day gross

EDA:
Plot change of gross in movies by the hour
Clustering
Rolling Average

Data would not be IID if we use all the hours
Take the deltas, use starting hour, and then deltas

If we narrow down the data to just opening day, how much is left? 
Overall #s, and # of Unique Films

Overall Data Set: 41k rows
Unique/Opening Day: 
Try to hit 2000 unique

If only using opening day data, how much data do I really have
Any missing data

Pull DP Plan for as many theatres as 
Predict region gross for an opening day

Models:
CLUSER DATA AND SEE IF IT SHOWS ANY NEW GROUPING
CAN PERFORM SEPARATE MODELS ON DIFFERENT CLUSTERS, ASK TOMAS HOW TO APPLY THAT


# Picture List
WIth empty data, try to figure out a way to impute some 
    ie. using RT to determine empty CS, inflation, create a secondary Genre field, setting avg run time by time period, add season
With Regression or Decision Trees, lookup how to weigh the decades (class weights for a model), so that later rows would be heavier
Can try a neural network

Could combine with Review data source
Movielens.com - movie recommendation site, might be opensource

# Capstone 3
Same Data, forecasting


Proposals:
Title
Description
Description of the Data
Need Features and Label Data

Predictive Modeling
Linear Regression
* Try different models
* Think about Inferential Modeling at some point
* Feature Engineering




Data:
2016 - 2020 Opening Day + Opening Weekend Hourly Grosses
+ Comparison Library (Genre, RT, Date, etc.)

Want to predict End of Day number when given an hour value
Can we predict all hours if given partial data?
    ex. have grosses up to 1pm, can we predict 2pm, 5pm, 11pm? Is that too intensive?

Methods:
How do I go about plotting multiple features?
What types of modelling will I need to implement?
Does this qualify for time series? what would this provide?
Mix of linear and logistical regression?
How to utilize Random Forrest/Gradient Descent/Neural Network?

Biggest factors could be similar Genre and similar hourly grosses

How to put together an ensemble model?

Create functions