# ece474-BayesianML
[Project 1 - Conjugate Priors](#project-1)

## Project 1 ##
The project is to implement conjugate estimators for the following scenarios: <br >
Binomial, <br >
Gaussian with known variance (i.e. estimate the mean), <br >
Gaussian with known mean (i.e. estimate the variance) <br >

You should plot the mean squared error both the ML and conjugate prior estimates, all on one plot, with a legend. For each scenario, choose 2-3 different values for the hyper parameters. 

Additionally, for at least one set of hyperparameter per scenario, plot the posterior density as it changes with observations. The easy way to do this is to just plot the pdf a few times, for different #s of observations.

Stretch Goal #1: Make a movie showing the posterior converge (2 points)
Stretch Goal #2: Implement the conjugate prior estimator for the unknown mean and variance case. Plot the pdf as it changes as in the first part. Note this will be a 3-Dim pdf, so use a heatmap or someting similar. (3 points)

## Project 2 - Linear Regression ##
Implement a basic linear regression and reproduce figures 3.7 and 3.8. You won't be able to reproduce them exactly since the draws of the observations are random, but it should be close.

Stretch Goal: Implement the evidence approximation and estimate alpha and beta (i.e. using eqns 3.92 and 3.95). 5 points

## Project 3 - Linear Classification ##
Generate a dataset for yourself consisting of 2 classes from a multivariate normal distribution. Mu1 = [1 1], Mu2 = [-1 -1], Sigma = eye(1). Implement two different linear classifiers on this data, a Gaussian generative model, and a logistic regression classifier. For the logistic regression classifier, you must implement the IRLS algorithm.

Report your % correct classification (it should be very high for this simple dataset). 

Plot a ROC curve for both of your classifiers, Plot the decision boundary for both your classifiers (hint, it should be a line)

Go to Kaggle or the UCI machine learning database, find a dataset that interests you, and try your classifiers on it. Make sure it is a binary dataset with only numerical features.

## Project 4 - Gaussian Processes ##
Re-do the second part of your linear regression assignment (i.e. the replication of figure 3.8) with Gaussian processes. This basically involves replacing your prediction equations (3.58 and 3.59) with the GP predictions (6.66 and 6.67). Use the Gaussian kernel (eqn 6.23) to build the covariance matrix C (eqn 6.62)

Stretch goal: Learn the hyperparemter (eqn 6.70) , (5 points)

## Project 5 - Expectation Maximization ##
Implement EM on a Gaussian mixture model in 1 and 2 dimensions. 

For the 1-D case, use K = 3, the choice of means, covariance and pi is up to you.  The algorithm is laid out explicitly in equations 9.23-9.28.

Produce a plot that shows a histogram of your generated observations, and overlay on that histogram the pdf you found. Plot this at algorithm init, and a couple other times as the algorithm converges. If you feel ambitious make a movie. If you want to see the algorithm break, artificially introduce a data point that exactly equals one of the means of the distribution.

For 2-D, create a plot similar to 9.8. Use the old faitfhul dataset and K = 2.

## Project 6 - Sampling Methods ##
The first part of the sampling methods project is to perform rejection sampling  to draw samples from a Gaussian mixture model. 

Use the same mixture model from the EM project (1-D only), and I suggest using regular normal RV for the proposal distribution. Draw samples using this method and plot a histogram of the generated samples against your GMM.

Part two of the sampling methods mini-project is to re-do your linear regression project using MCMC to find an estimate for the weights.

Reuse your project 2 to generate the same training data. Just do this for 25 training samples

Use Equation 3.10 as the likelihood function, to be used with the training samples you generated. You may select any distribution you want for the prior on the weights, and recall that the posterior density on the weights w is proportional to the likelihood x prior

Use the Metropolis algorithm as defined in equation 11.33 to compute an estimate of the weights

Stretch goal, 5 points:  Take MCMC all the way and draw samples from the predictive distribution as well.

## Project 7 - Sampling Using PyMC3 ##
