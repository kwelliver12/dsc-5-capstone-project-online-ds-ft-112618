
# NCAA Men's Basketball: What leads to wins?

When watching the NCAA basketball tournament, I was inspired to see what really had the biggest impact on games. Growing up a basketball player, I was always told offense wins games, but defense wins championships and I couldn’t help but wonder, “does this still hold true?”. That when I decided I wanted take a dataset and run it to find out which factors had the biggest impacts on the outcome of a basketball game. Being March, I decided to focus on NCAA Men’s Basketball. I obtained a dataset of every game dating back to 2013 to use for training and testing. I started off by cleaning up the data and eliminating unnecessary features. 

Print and sum null values to see what columns have significant data missing. Drop columns with a significant number of null values and drop the rows containing null values in columns with low number of null values.

For this project, only Division 1 data should be considered. Check to see how many unique values are in the division column and delete the all but division 1 values.

In order to focus one what impacts wins and losses we must create a win/loss feature. This will be done by subtracting the number of away points from the number of home points. If the value is possitive (greater than zero) this means the home team won the game. If the value is negative, the home team lost the game i.e. the away team won.  

In order to perform further testing, the win/loss column will be represented by a 0 or 1 rather than a W or L. For the sake of testing, the tests will be from the home team's point of view, meaning a 1 will represent a win for the home team and a 0 will respresent a loss for the home team.

Now that we have cleaned our dataframe we will want to save it before making any more changes.

# Creating Offensive and Defensive tables

The table below is created to contain all categories related to offense for the home team. I decided to use the three point, two points, and free throws percentages because it is a more descriptive value that total shots taken or made. I also chose to not include field goal percentage because it is composed of two and three point shots. I thought it would be better for training and tetsing to use the two categories individually.

The table below is created to contain the categories related to defense for the home team.

Use a heatmap to plot the correlation to help see the relationships better visually.

Ran a correlation for the offensive values to see what is related. As you can see, points is correlated with three point percentage, two point percentage, as well as assists. This is not surpsing because all these lead to points. These values, threes, twos, and assits are also all relatively correlated wirh eachother.

As you can see from the corrlation above, the defensive categories are not strongly related.

# Regression Analysis
Use logistic regression to describe the data and explain the relationship between the target (win/loss) and the independent features.

#### Offense 

Below we will normalize the data because the features all have different values, which can lead to skewed results. We will then train/test the data to run the logistic regression.

With this data set we use the accuracy to tets our results. We are using accuracy because we are testing to see if our prediction is either right or wrong. There is no penalty for a prediction being wrong, which could be the case regarding medical testing.

Our model is about 71% accurate when testing our data.


```python
plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) #Create the basic matrix.

plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x1a1a243550>




![png](NCAA-read_files/NCAA-read_18_1.png)


The confusion matrix shows us thst our model accurately predicts a win when the home team wins a large number of times. Comparatively there is a low number of predictions of a loss when a team looses. This could mean that our data and tests are a better predictive of wins than they are of losses.

Both original and normalized confusion matrices show that our accuracy is high when predicting a win accurately. 

#### Defense

Must normalize the data so the features can be compared equally.

Given the defensive data, the model accurately predicts the correct outcome about 68% of the time.


```python
plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) #Create the basic matrix.

plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x1a1beee898>




![png](NCAA-read_files/NCAA-read_24_1.png)


Like the confusion matrix for offense values, the defense can accurately contribute to selecting when a team wins a high number of times. This also shows the it is poor when predicting a loss.

# Offensive Score


Below we will use feature engineering to create a new feature. This will be offensive and defensive score. The offense and defensive score categories will weight the importance of features within the offense and defense tables. We will use this new weighted score to see if/how it is a good predictor of the outcome of a game.

I weighted the offensive features combined to create a new one. Turnovers is also included negatively because this impacts the offense in a negative way. I decided to use turnovers as a negative for the offense because a lot of times a turnover results from a bad offensive play versus a positive play for the defense.

Plot histograms of the offensive features to see the distribution

Find minimum, maximum, median, and mean values of the offensive score feature to better understand the values. 
The home offensive score mean is 78.77
The home offensive score min is 37.95
The home offensive score median is 78.91
The home offensive score max is 119.93


# Defensive Score

Below is the newly created defensive score feature. This new feature combines the weighted values of the defense tbale.

Like for the offensive score, we found the mean, median, min, and max for defensive score: 
The home deffensive score mean is 76.35
The home deffensive score min is 23.5
The home deffensive score median is 75.8
The home deffensive score max is 144.2

# What features are important factors in win?

Below we will train and test our offensive and defensive data. We will use the trained data to create a decision tree classifier to see what features are important and contributing factors to predict the outcome of a game.


```python
def plot_feature_importances(model):
    n_features = offense_h_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), offense_h_train.columns.values)
    plt.xlabel('Features importance')
    plt.ylabel('feature')
plot_feature_importances(tree_clf)    
```


![png](NCAA-read_files/NCAA-read_36_0.png)


The graph above shows that our value of offensive score is a good predictor of what will win a basketball game.


```python
def plot_feature_importances(model):
    n_features = defense_h_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), defense_h_train.columns.values)
    plt.xlabel('Features importance')
    plt.ylabel('feature')
plot_feature_importances(tree_clf)    
```


![png](NCAA-read_files/NCAA-read_38_0.png)


The above graph shows our defensive score is a good predictor for the outcome of a game.
Below we will drop the defense score to see how individual categories predict outcome of a game.

#### Drop offense score
Below we dropped the offense score to see what individual categories impact the outcome of a game


```python
def plot_feature_importances(model):
    n_features = A_oh_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), A_oh_train.columns.values)
    plt.xlabel('Features importance')
    plt.ylabel('feature')
plot_feature_importances(tree_clf)    
```


![png](NCAA-read_files/NCAA-read_41_0.png)


As you can see points is the biggest factor in predicting the outcome of a game. This is not surprsing because it is points specifically that tells you whether you win or loose. To have a better understanding of the factors contributing to a win, the points feature will also be removed.

The above graph shows how assists impacts the outcome of a game. When thinking about this further, we see how this makes sense. An assist is recorded when a player passes the ball to their teammate and he/she shots and makes a basket. This leads to points, which as we know leads to wins. We will run this one more time, removing the assits feature.

When removing features we see that two point percentage has the highest importance for predicting the outcome of a game, followed by three point percentage. 

# Is there home court advantage?

The tests above show that the home team wins 68% of the time. This shows that there is some home court advatage, but 68% seems rather high. This could be affected by tournament games and neutral sites. Thhese features were removed from the data set because they had a large amount of null values. One way these fetaures could increase home winning percentage is often in tournament games the higher seed is considered the home team.

# How many points do you need to win?
First find the mean points scored by both teams, home and away.

We found that the average score for the home team is 76 points and for the away team it is 69 points. It is not surprising the mean points scored for the home team is higher than away given that we found the home team wins about 68.5% of the time. We will continue running tests for the home team.

After testing, I found that if the home team scores 75 or more points, they will win 85% of the time.


```python
win = w.h_points
loss = l.h_points
plt.hist(win, alpha=0.5, bins=30, align='mid', label='Win', color='green')
plt.hist(loss, alpha=0.5, bins=30, align='mid', label='Loss', color='blue')
plt.xlim(20,140)
plt.locator_params(axis='x', nbins=12)
plt.xlabel('Points per Game', fontsize=12)
plt.ylabel('Total Games', fontsize=12)
plt.legend()
plt.title('Points per Game leading to a Win/Loss', fontsize=14);
```


![png](NCAA-read_files/NCAA-read_50_0.png)


# What 2pt% do you need to win the game???


```python
h_two_pct_mean = ncaa.h_two_points_pct.mean()
h_two_pct_mean
```




    51.80897063778587




```python
two_pct_game = ncaa[ncaa['h_two_points_pct'] >= 60]
two_pct_win = two_pct_game[two_pct_game['win_h'] == 1.0]
```


```python
print(two_pct_game.shape)
print(two_pct_win.shape)
```

    (844, 91)
    (746, 91)


Start by finding the average two point score percentage, which is 51.8%. Decided to increase two point percentage to 60% to see how this would predict the outcome and found that team shooting 60% from the two point range will win 88% of the time.


```python
win = w.h_two_points_pct
loss = l.h_two_points_pct
plt.hist(win, alpha=0.5, bins=30, align='mid', label='Win', color='green')
plt.hist(loss, alpha=0.5, bins=30, align='mid', label='Loss', color='blue')
plt.xlim(0,100)
plt.locator_params(axis='x', nbins=12)
plt.xlabel('Two Point Pct for Game', fontsize=14)
plt.ylabel('Total Games', fontsize=14)
plt.legend()
plt.title('Two Point Pct for Game vs Win/Loss', fontsize=16);
```


![png](NCAA-read_files/NCAA-read_56_0.png)


# What 3pt% do you need to win the game???

Find the average of three points made during the game, which is 36%. Decided to run test to see how well a team did when shooting 45% from three point range. Found that when teams shoot 45% from three point range they win 89% of the time.


```python
win = w.h_three_points_pct
loss = l.h_three_points_pct
plt.hist(win, alpha=0.5, bins=30, align='mid', label='Win', color='green')
plt.hist(loss, alpha=0.5, bins=30, align='mid', label='Loss', color='blue')
plt.xlim(0,100)
plt.locator_params(axis='x', nbins=12)
plt.xlabel('Three Point Pct for Game', fontsize=14)
plt.ylabel('Total Games', fontsize=14)
plt.legend()
plt.title('Three Point Pct for Game vs Win/Loss', fontsize=16);
```


![png](NCAA-read_files/NCAA-read_59_0.png)



```python

```
