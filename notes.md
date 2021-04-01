# Capstone 2
CPI Library comes from 
CPI Home : U.S. Bureau of Labor Statisticshttps://www.bls.gov › cpi


Models:
Might have to do models without TM/CS, and one with
Linear Regression
Tree Based Models next
Start basic: leave blank
Then impute the Mean first as your imputation
add Features, then build from there

show distribution of dom gross by studio, # of films, as well as mean, proportion of revenue

whats going on with small opening weekend numbers but larger domestic gross
what is about these movies that did well? besides just the opening weekend
group smaller distributors into one, or cluster and replace distributor
Cluster twice for genre
Try one and see what it gives, use those clusters as features, and try a linear regression
Standardize, do a LASSO Regression will do that same
    FOUND ROBUST SCALING TO HANDLE OUTLIERS
    Plot those features on a scatter matrix
    do some EDA on the features, talk about what features were important
Random Forest
    Talk about feature importance 

Question on modeling:
Baseline used the entire DF, but other models don't use the entire DF, is that an issue? Should they all be the same?
    WHy use Opening Weekend as the first one? 
    Generate Correlation Matrix with Target
    df.corr, gives correlation values of anything related within the data frame, use continuous/numeric data with pearson correlation, can create a heatmap
    may need to switch the correlation for categorical features
    Looping by GroupBy

Do another model on the smaller data set with more scores
Standardize data and normalize
Stats Model has more inferential stats, have to add bias/intercept manually
    Can fix with add constant
    sklearn coeffecients

At what stage should I be scaling/regularizing?
Where should I go with this Linear regression? I have one feature in, do I just dump them all? What kind of EDA should I do beforehand?


Imputing values when we don't have the data available or any data avilable to predict it? Should I also be modeling these factors?
    Use EDA to determine feature selection
After I clean up the NaN's, which model would also be best given this data: Decision Trees, Random Forest? What are the use cases for the others? (Gradient Boosting)
    Yes, Gradient Boosting, could even do Neural Net with Relu
Could I incorporate any Clustering 

* Need to Impute on Dom Locs, RT, CS for historical data



Process:
Dropped rows with Empty Dom Gross
Set Release Date as Date Time Object, parsed out Year Month, and Week Num
Applied 2D impuation
Remove $$ from gross fields, and converted to float
Created runtime_calc function to convert runtime from H:MM into MMM
Split Genre, only looking at Primary Genre
Inflated Grosses with inflation calculator
Convert CS into a numeric scale
After Creating Dummy Variables for Dist, Format, Rtg, Genre, Pattern, now have 489 Features


CLUSER DATA AND SEE IF IT SHOWS ANY NEW GROUPING
CAN PERFORM SEPARATE MODELS ON DIFFERENT CLUSTERS, ASK TOMAS HOW TO APPLY THAT
PCA AS PART OF EDA?
RF or GRADIENT BOOSTING FOR PREDICTIVE VALUES
LINEAR REGRESSION FOR INFERENTIAL VALUES, AND KNOWING WHAT COEFFECIENTS ARE THE MOST IMPACTFUL

Methods:
How do I go about plotting multiple features?
What types of modelling will I need to implement?
Does this qualify for time series? what would this provide?
Mix of linear and logistical regression?
How to utilize Random Forrest/Gradient Descent/Neural Network?


## Rubric:
Github:
Py files, if you're copying and pasting a block of code, it should probably be a function

Presentation:





# Picture List
WIth empty data, try to figure out a way to impute some 
    ie. using RT to determine empty CS, inflation, create a secondary Genre field, setting avg run time by time period, add season
With Regression or Decision Trees, lookup how to weigh the decades (class weights for a model), so that later rows would be heavier
Can try a neural network

Could combine with Review data source
Movielens.com - movie recommendation site, might be opensource



# Capstone 3
Same Data, forecasting

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



Biggest factors could be similar Genre and similar hourly grosses

How to put together an ensemble model?

Create functions