---
title: "Monitor the training process"
teaching: 0
exercises: 0
questions:
- tbd
objectives:
- "Explain the importance of splitting the training data"
- "Use the data splits to plot the training process"
- "Measure the performance of your deep neural network"
- "Interpret the training plots to recognize overfitting"
- "Implement basic strategies to prevent overfitting"
- "Understand the effect of regularization techniques"

keypoints:
- "Separate training, validation, and test sets allows monitoring and evaluating your model."
- "Dropout is a way to prevent overfitting"
---

## Import dataset
Here we want to work with the *weather prediction dataset* which can be downloaded from [ADD ZENODO LINK].
It contains daily weather observations from 18 different European cities or places through the years 2000 to 2010. For all locations the data contains the variables ‘mean temperature’, ‘max temperature’, and ‘min temperature’. In addition, for multiple of the following variables are provided: 'cloud_cover', 'wind_speed', 'wind_gust', 'humidity', 'pressure', 'global_radiation', 'precipitation', 'sunshine', but not all of them are provided for all locations. A more extensive description of the dataset including the different physical units is given in accompanying metadata file.
![18 locations in the weather prediction dataset](../fig/03_weather_prediction_dataset_map.png)

~~~
filename_data = os.path.join(path_data, "weather_prediction_dataset.csv")
data = pd.read_csv(filename_data)
data.head()
~~~
{:.language-python}

### Select a subset and split into data (X) and labels (y)
The full dataset comprises 10 years (3654 days) from which we here will only select the first 3 years.
In addition, we will remove all columns of the place that we want to make predictions on (here: Düsseldorf which is about in the middle of all 18 locations).

~~~
columns_selected = [x for x in data.columns if not x.startswith("DUSSELDORF") if x not in ["DATE", "MONTH"]]
X_data = data.loc[:365*3][columns_selected]
X_data.head()
~~~
{:.language-python}


| | DATE 	| MONTH | 	BASEL_cloud_cover 	| 	BASEL_humidity 	| 	BASEL_pressure	| ... |
|------:|------:|---------------:|--------------:|------------------:|------------:|------------:|
|0| 	20000101 	|1 	|8 	|0.89 	|1.0286|... |
|1| 	20000102 	|1 	|8 	|0.87 	|1.0318|... |
|2| 	20000103 	|1 	|5 	|0.81 	|1.0314|... |
|3| 	20000104 	|1 	|7 	|0.79 	|1.0262|... |
|4| 	20000105 	|1 	|5 	|0.90 	|1.0246|... |
{: .output}

As a label, that is the the values we want to later predict, we here pick the sunshine hours which we can get by
~~~
y_data = data.loc[:365*3]["DUSSELDORF_sunshine"].values
~~~
{:.language-python}

### Split data and labels into training, validation, and test set
As with classical machine learning techniques, it is common in deep learning to split off a *test set* which remains untouched during model training and tuning. It is then later be used to evaluate the model performance. Here, we will also split off an additional *validation set*, the reason of which will hopefully become clearer later in this lesson.

## Regression and classification - how to set a training goal
- Explain how to define the output part of a neural network
- What is the loss function (and which one to chose for a regression or classification task)?


In episode 2 we trained a dense neural network on a *classification task*. For this one hot encoding was used together with a Categorical Crossentropy loss function.
This measured how close the distribution of the neural network outputs corresponds to the distribution of the three values in the one hot encoding.
Now we want to work on a *regression task*, thus not prediciting the right class for a datapoint but a certain value (could in principle also be several values). In our example we want to predict the sunshine hours in Düsseldorf (or any other place in the dataset) for a particular day based on the weather data of all other places. 

### Network output layer:
The network should hence output a single float value which is why the last layer of our network will only consist of a single node. 

> ## Create the neural network
>
> We have seen how to build a dense neural network in episode 2. 
> Try now to construct a dense neural network with 3 layers for a regression task.
> You could for instance start with a network of a dense layer with 100 nodes, followed by one with 50 nodes and finally an output layer.
>
> * What must here be the dimension of our input layer?
> * How would our output layer look like? What about the activation function?
>
> > ## Solution
> > ~~~
> > def create_nn(n_features, n_predictions):
> >     # Input layer
> >     input = Input(shape=(n_features,), name='input')
> > 
> >     # Dense layers
> >     layers_dense = Dense(100, 'relu')(input)
> >     layers_dense = Dense(50, 'relu')(layers_dense)
> > 
> >     # Output layer
> >     output = Dense(n_predictions)(layers_dense)
> > 
> >     return Model(inputs=input, outputs=output, name="weather_prediction_model")
> > 
> > model = create_nn(n_features=X_data.shape[1], n_predictions=1)
> > model.summary()
> > ~~~
> > {:.language-python}
> >
> > ~~~
> > Model: "weather_prediction_model"
> > _________________________________________________________________
> > Layer (type)                 Output Shape              Param #   
> > =================================================================
> > input (InputLayer)           [(None, 152)]             0         
> > _________________________________________________________________
> > dense_0 (Dense)              (None, 100)               15300     
> > _________________________________________________________________
> > dense_1 (Dense)              (None, 50)                5050      
> > _________________________________________________________________
> > dense_2 (Dense)              (None, 1)                 51        
> > =================================================================
> > Total params: 20,401
> > Trainable params: 20,401
> > Non-trainable params: 0
> > _________________________________________________________________
> > ~~~
> > {:.output}
> >
> > The shape of the input layer has to correspond to the number of features in our data: 152
> > 
> > The output layer here is a dense layer with only 1 node. And we here have chosen to use *no activation function*.
> > While we might use *softmax* for a classification task, here we do not want to restrict the possible outcomes for a start.
> > 
> > In addition, we have here chosen to write the network creation as a function so that we can use it later again to initiate new models.
> {:.solution}
{:.challenge}

When compiling the model we can define a few very important aspects.

### Loss function:
The loss is what the neural network will be optimized on during training, so chosing a suitable loss function is crucial for training neural networks.
In the given case we want to stimulate that the prodicted values are as close as possible to the true values. This is commonly done by using the *mean squared error* (mse) or the *mean absolute error* (mae), both of which should work OK in this case. Often, mse is prefered over mae because it "punishes" large prediction errors more severely.
In keras this is implemented in the `keras.losses.MeanSquaredError` class.

### Optimizer:
Somewhat coupled to the loss function is the *optimizer* that we want to use. 
The *optimizer* here refers to the algorithm with which the model learns to optimize on the set loss function. A basic example for such an optimizer would be *stochastic gradient descent*. For now, we can largely skip this step and simply pick one of the most common optimizers that works well for most tasks: the *Adam optimizer*. 

### Metrics:
In our first example (episode 2) we plotted the progression of the loss during training. 
That is indeed a good first indicator if things are working alright, i.e. if the loss is indeed decreasing as it should. 
However, when models become more complicated then also the loss functions often become less intuitive (side remark: e.g. when adding L1 or L2 regularization). 
That is why it is good practice to monitor the training process with additional, more intuitive metrics. 
They are not used to optimize the model, but are simply recorded during training. 
With Keras they can simply be added via `metrics=[...]` and can contain one or multiple metrics of interest. 
Here we could for instance chose to use `'mae'` the mean absolute error, or the the *root mean squared error* (RMSE) which unlike the *mse* has the same units as the predicted values. Finally, after compiling we train the model on our training data for 200 epochs.

~~~
model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
~~~
{: .language-python}

## Train a dense neural network
Now that we created and compiled our dense neural network, we can start training it.
~~~
history = model.fit(X_train, y_train,
                    batch_size=10,
                    epochs=200,
                    verbose=2)
~~~
{: .language-python}

We can plot the training process using the history:
~~~
history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df['root_mean_squared_error'])
plt.xlabel("epochs")
plt.ylabel("RMSE")
~~~
{: .language-python}
![Output of plotting sample](../fig/03_training_history_1_rmse.png)

This looks very promising! Our loss ("mse") is dropping nicely and while it maybe keeps fluctuating a bit it does end up at fairly low *mse* values.
But the *mse* is just the *mean* squared error, so we might want to look a bit more in detail how well our just trained model does in predicting the sunshine hours.

## Evaluate our model
There is not a single way to evaluate how a model performs. But there is at least two very common approaches. For a *classification task* that is to compute a *confusion matrix* for the test set which shows how often particular classes were predicted correctly or incorrectly. For the present *regression task* however, it makes more sense to compare true and predicted values in simple scatter plot.

First, we will do the actual prediction step. 
> ## Predict the labels for both training and test set and compare to the true values
> Even though we here use a different model architecture and a different task compared to episode 2, the prediction step is mostly identical.
> Here you should predict the labels for the training set and the test set and then compare them in a scatter plot to the true labels.
> 
> * Is the accuracy of the predictions as you expected (or better/worse)? 
> * Is there a noteable difference between training set and test set? And if so, any idea why?
> > ~~~
> > y_train_predicted = model.predict(X_train)
> > y_test_predicted = model.predict(X_test)
> > ~~~
> > {: .language-python}
> > We can then compare those to the true labels, for instance by
> > ~~~
> > fig, axes = plt.subplots(1, 2, figsize=(12, 6))
> > plt.style.use('ggplot')  # optional, that's only to define a visual style
> > axes[0].scatter(y_train_predicted, y_train, s=10, alpha=0.5, color="teal")
> > axes[0].set_title("training set")
> > axes[0].set_xlabel("predicted sunshine hours")
> > axes[0].set_ylabel("true sunshine hours")
> > 
> > axes[1].scatter(y_test_predicted, y_test, s=10, alpha=0.5, color="teal")
> > axes[1].set_title("test set")
> > axes[1].set_xlabel("predicted sunshine hours")
> > axes[1].set_ylabel("true sunshine hours")
> > ~~~
> > {: .language-python}
> > ![Scatter plot to evaluate training and test set](../fig/03_regression_compare_training_and_test_performance.png)
> > Maybe that is not exactly what you expected? What is the issue here? Any ideas?
> > 
> > The accuracy on the training set is fairly good. 
> > In fact, considering that the task of predicting the daily sunshine hours is really not easy it might even be surprising how well the model predicts that 
> > (at least on the training set). Maybe a little too good?
> > For those familiar with (classical) machine learning this might look familiar. 
> > It is a very clear signature of *overfitting* which means that the model has to some extend memorized aspects of the training data. 
> > As a result makes much more accurate predictions on the training data than on unseen data.
> {:.solution}
{:.challenge}

Overfitting also happens in classical machine learning, but there it is usually interpreted as the model having more parameters than the training data would justify (say, a decision tree with too many branches for the number of training instances). As a consequence one would reduce the number of parameters to avoid overfitting.
In deep learning the situation is slightly different. It can -same as for classical machine learning- also be a sign of having a *too big* model, meaning a model with too many parameters (layers and/or nodes). However, in deep learning higher number of model parameters are often still considered acceptable and models often perform best (in terms of prediction accuracy) when they are at the verge of overfitting. So, in a way, training deep learning models is always a bit like playing with fire...

## Watch your model training closely
As we just saw, deep learning models are prone to overfitting. Instead of iterating through countless cycles of model trainings and subsequent evaluations with a reserved test set, it is common practice to work with a 2nd split off dataset to monitor the model during training. This is the *validation set* which can be regarded as a 2nd test set. As with the test set the datapoints of the *validation set* are not used for the actual model training itself. Instead we evalute the model with the *validation set* after every epoch during training, for instance to splot if we see signs of clear overfitting.

Let's give this a try!

We need to initiate a new model -- otherwise Keras will simply assume that we want to continue training the model we already trained above.
~~~
model = create_nn(n_features=X_data.shape[1], n_predictions=1)
model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
~~~
{: .language-python}

But now we train it with the small addition of also passing it our validation set:
~~~
history = model.fit(X_train, y_train,
                    batch_size=10,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    verbose=2)
~~~
{: .language-python}

As before the history allows plotting the training progress.
~~~
history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
~~~
{: .language-python}
![Output of plotting sample](../fig/03_training_history_2_rmse.png)

This clearly shows that something is not completely right here. 
The model predictions on the validation set quickly seem to reach a plateau while the performance on the training set keeps improving.
That is a clear signature of overfitting.
