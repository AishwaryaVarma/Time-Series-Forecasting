# Time-Series-Forecasting


Time series forecasting is the task of predicting future data points, given some historical data. It is commonly used in a variety of tasks from weather forecasting, retail and sales forecasting, stock market prediction, and in behavior prediction (such as predicting the flow of car traffic over a day).



Time Series Forecasting
A time series is data collected periodically, over time. Time series forecasting is the task of predicting future data points, given some historical data. It is commonly used in a variety of tasks from weather forecasting, retail and sales forecasting, stock market prediction, and in behavior prediction (such as predicting the flow of car traffic over a day). There is a lot of time series data out there, and recognizing patterns in that data is an active area of machine learning research!

<img src='notebook_ims/time_series_examples.png' width=80% />


Employ using DeepAR


DeepAR utilizes a recurrent neural network (RNN), which is designed to accept some sequence of data points as historical input and produce a predicted sequence of points. So, how does this model learn?

During training, we will have to provide a training dataset (made of several time series) to a DeepAR estimator. The estimator looks at all the training time series and tries to identify similarities across them. It trains by randomly sampling training examples from the training time series.

Each training example consists of a pair of adjacent context and prediction windows of fixed, predefined lengths.
The context_length parameter controls how far in the past the model can see.
The prediction_length parameter controls how far in the future predictions can be made.


Since DeepAR trains on several time series, it is well suited for data that exhibit recurring patterns.


Energy Consumption Data

The data we'll be working with in this notebook is data about household electric power consumption, over the globe. The dataset is originally taken from Kaggle, and represents power consumption collected over several years from 2006 to 2010. With such a large dataset, we can aim to predict over long periods of time, over days, weeks or months of time. Predicting energy consumption can be a useful task for a variety of reasons including determining seasonal prices for power consumption and efficiently delivering power to people, according to their predicted usage.


Machine Learning Workflow
This notebook approaches time series forecasting in a number of steps:

Loading and exploring the data
Creating training and test sets of time series
Formatting data as JSON files and uploading to S3
Instantiating and training a DeepAR estimator
Deploying a model and creating a predictor
Evaluating the predictor
Let's start by loading in the usual resources.

In [1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
Load and Explore the Data
We'll be loading in some data about global energy consumption, collected over a few years. The below cell downloads and unzips this data, giving you one text file of data, household_power_consumption.txt.

In [2]:
# ! wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/March/5c88a3f1_household-electric-power-consumption/household-electric-power-consumption.zip
# ! unzip household-electric-power-consumption
Read in the .txt File
The next cell displays the first few lines in the text file, so we can see how it is formatted.

Pre-Process the Data
The 'household_power_consumption.txt' file has the following attributes:

Each data point has a date and time (hour:minute:second) of recording
The various data features are separated by semicolons (;)
Some values are 'nan' or '?', and we'll treat these both as NaN values
Managing NaN values


Below, I'm reading the file in as a DataFrame and filling NaN values with feature-level averages.

In [4]:
import txt_preprocessing as pprocess

# create df from text file
initial_df = pprocess.create_df('household_power_consumption.txt', sep=';')

# fill NaN column values with *average* column value
df = pprocess.fill_nan_with_mean(initial_df)

# print some stats about the data
print('Data shape: ', df.shape)
df.head()
Data shape:  (2075259, 7)
Out[4]:
Global_active_power	Global_reactive_power	Voltage	Global_intensity	Sub_metering_1	Sub_metering_2	Sub_metering_3
Date-Time							
2006-12-16 17:24:00	4.216	0.418	234.84	18.4	0.0	1.0	17.0
2006-12-16 17:25:00	5.360	0.436	233.63	23.0	0.0	1.0	16.0
2006-12-16 17:26:00	5.374	0.498	233.29	23.0	0.0	2.0	17.0
2006-12-16 17:27:00	5.388	0.502	233.74	23.0	0.0	1.0	17.0
2006-12-16 17:28:00	3.666	0.528	235.68	15.8	0.0	1.0	17.0
Global Active Power
 

In [5]:
# Select Global active power data
power_df = df['Global_active_power'].copy()
print(power_df.shape)
(2075259,)
In [6]:
# display the data 
plt.figure(figsize=(12,6))
# all data points
power_df.plot(title='Global active power', color='blue') 
plt.show()

Since the data is recorded each minute, the above plot contains a lot of values. So, I'm also showing just a slice of data, below.

In [7]:
# can plot a slice of hourly data
end_mins = 1440 # 1440 mins = 1 day

plt.figure(figsize=(12,6))
power_df[0:end_mins].plot(title='Global active power, over one day', color='blue') 
plt.show()

Hourly vs Daily
There is a lot of data, collected every minute, and so I could go one of two ways with my analysis:


In [8]:
# resample over day (D)
freq = 'D'
# calculate the mean active power for a day
mean_power_df = power_df.resample(freq).mean()

# display the mean values
plt.figure(figsize=(15,8))
mean_power_df.plot(title='Global active power, mean per day', color='blue') 
plt.tight_layout()
plt.show()

In this plot, we can see that there are some interesting trends that occur over each year. It seems that there are spikes of energy consumption around the end/beginning of each year, which correspond with heat and light usage being higher in winter months. We also see a dip in usage around August, when global temperatures are typically higher.


My goal will be to take full years of data, from 2007-2009, and see if I can use it to accurately predict the average Global active power usage for the next several months in 2010!



All of the time series will start at the same time point t_start (or t0).
When preparing data, it's important to use a consistent start point for each time series; DeepAR uses this time-point as a frame of reference, which enables it to learn recurrent patterns e.g. that weekdays behave differently from weekends or that Summer is different than Winter.
You can change the start and end indices to define any time series you create.
We should account for leap years, like 2008, in the creation of time series.
Generally, we create Series by getting the relevant global consumption data (from the DataFrame) and date indices.
# get global consumption data
data = mean_power_df[start_idx:end_idx]

# create time series for the year
index = pd.DatetimeIndex(start=t_start, end=t_end, freq='D')
time_series.append(pd.Series(data=data, index=index))
In [9]:
def make_time_series(mean_power_df, years, freq='D', start_idx=16):
    '''Creates as many time series as there are complete years. This code
       accounts for the leap year, 2008.
      :param mean_power_df: A dataframe of global power consumption, averaged by day.
          This dataframe should also be indexed by a datetime.
      :param years: A list of years to make time series out of, ex. ['2007', '2008'].
      :param freq: The frequency of data recording (D = daily)
      :param start_idx: The starting dataframe index of the first point in the first time series.
          The default, 16, points to '2017-01-01'. 
      :return: A list of pd.Series(), time series data.
      '''
    
    # store time series
    time_series = []
    
    # store leap year in this dataset
    leap = '2008'

    # create time series for each year in years
    for i in range(len(years)):

        year = years[i]
        if(year == leap):
            end_idx = start_idx+366
        else:
            end_idx = start_idx+365

        # create start and end datetimes
        t_start = year + '-01-01' # Jan 1st of each year = t_start
        t_end = year + '-12-31' # Dec 31st = t_end

        # get global consumption data
        data = mean_power_df[start_idx:end_idx]

        # create time series for the year
        index = pd.DatetimeIndex(start=t_start, end=t_end, freq=freq)
        time_series.append(pd.Series(data=data, index=index))
        
        start_idx = end_idx
    
    # return list of time series
    return time_series
Test the results

# test out the code above

# yearly time series for our three complete years
full_years = ['2007', '2008', '2009']
freq='D' # daily recordings

# make time series
time_series = make_time_series(mean_power_df, full_years, freq=freq)
In [11]:
# display first time series
time_series_idx = 0

plt.figure(figsize=(12,6))
time_series[time_series_idx].plot()
plt.show()

Splitting in Time


In [12]:
# create truncated, training time series
def create_training_series(complete_time_series, prediction_length):
    '''Given a complete list of time series data, create training time series.
       :param complete_time_series: A list of all complete time series.
       :param prediction_length: The number of points we want to predict.
       :return: A list of training time series.
       '''
    # get training series
    time_series_training = []
    
    for ts in complete_time_series:
        # truncate trailing 30 pts
        time_series_training.append(ts[:-prediction_length])
        
    return time_series_training
In [13]:
# test your code!

# set prediction length
prediction_length = 30 # 30 days ~ a month

time_series_training = create_training_series(time_series, prediction_length)
Training and Test Series
We can visualize what these series look like, by plotting the train/test series on the same axis. We should see that the test series contains all of our data in a year, and a training series contains all but the last prediction_length points.

In [14]:
# display train/test time series
time_series_idx = 0

plt.figure(figsize=(15,8))
# test data is the whole time series
time_series[time_series_idx].plot(label='test', lw=3)
# train data is all but the last prediction pts
time_series_training[time_series_idx].plot(label='train', ls=':', lw=3)

plt.legend()
plt.show()

Convert to JSON
According to the DeepAR documentation, DeepAR expects to see input training data in a JSON format, with the following fields:

start: A string that defines the starting date of the time series, with the format 'YYYY-MM-DD HH:MM:SS'.
target: An array of numerical values that represent the time series.
cat (optional): A numerical array of categorical features that can be used to encode the groups that the record belongs to. This is useful for finding models per class of item, such as in retail sales, where you might have {'shoes', 'jackets', 'pants'} encoded as categories {0, 1, 2}.
The input data should be formatted with one time series per line in a JSON file. Each line looks a bit like a dictionary, for example:

{"start":'2007-01-01 00:00:00', "target": [2.54, 6.3, ...], "cat": [1]}
{"start": "2012-01-30 00:00:00", "target": [1.0, -5.0, ...], "cat": [0]} 
...

The starting date, "start," will be the index of the first row in a time series, Jan. 1st of that year.
The "target" will be all of the energy consumption values that our time series holds.
We will not use the optional "cat" field.
Complete the following utility function, which should convert pandas.Series objects into the appropriate JSON strings that DeepAR can consume.

In [15]:
def series_to_json_obj(ts):
    '''Returns a dictionary of values in DeepAR, JSON format.
       :param ts: A single time series.
       :return: A dictionary of values with "start" and "target" keys.
       '''
    # get start time and target from the time series, ts
    json_obj = {"start": str(ts.index[0]), "target": list(ts)}
    return json_obj
In [16]:
# test out the code
ts = time_series[0]

json_obj = series_to_json_obj(ts)

#print(json_obj)
Saving Data, Locally
The next helper function will write one series to a single JSON line, using the new line character '\n'. The data is also encoded and written to a filename that we specify.

In [17]:
# import json for formatting data
import json
import os # and os for saving

def write_json_dataset(time_series, filename): 
    with open(filename, 'wb') as f:
        # for each of our times series, there is one JSON line
        for ts in time_series:
            json_line = json.dumps(series_to_json_obj(ts)) + '\n'
            json_line = json_line.encode('utf-8')
            f.write(json_line)
    print(filename + ' saved.')
In [18]:
# save this data to a local directory
data_dir = 'json_energy_data'

# make data dir, if it does not exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
In [19]:
# directories to save train/test data
train_key = os.path.join(data_dir, 'train.json')
test_key = os.path.join(data_dir, 'test.json')

# write train/test JSON files
write_json_dataset(time_series_training, train_key)        
write_json_dataset(time_series, test_key)
json_energy_data/train.json saved.
json_energy_data/test.json saved.
Uploading Data to S3
Next, to make this data accessible to an estimator, I'll upload it to S3.

Sagemaker resources
Let's start by specifying:

The sagemaker role and session for training a model.
A default S3 bucket where we can save our training, test, and model data.
In [20]:
import boto3
import sagemaker
from sagemaker import get_execution_role
In [21]:
# session, role, bucket
sagemaker_session = sagemaker.Session()
role = get_execution_role()

bucket = sagemaker_session.default_bucket()
EXERCISE: Upoad both training and test JSON files to S3
Specify unique train and test prefixes that define the location of that data in S3.

Upload training data to a location in S3, and save that location to train_path
Upload test data to a location in S3, and save that location to test_path
In [22]:
# general prefix
prefix='deepar-energy-consumption'

# *unique* train/test prefixes
train_prefix   = '{}/{}'.format(prefix, 'train')
test_prefix    = '{}/{}'.format(prefix, 'test')

# uploading data to S3, and saving locations
train_path  = sagemaker_session.upload_data(train_key, bucket=bucket, key_prefix=train_prefix)
test_path   = sagemaker_session.upload_data(test_key,  bucket=bucket, key_prefix=test_prefix)
In [23]:
# check locations
print('Training data is stored in: '+ train_path)
print('Test data is stored in: '+ test_path)
Training data is stored in: s3://sagemaker-us-west-1-467380521728/deepar-energy-consumption/train/train.json
Test data is stored in: s3://sagemaker-us-west-1-467380521728/deepar-energy-consumption/test/test.json
Training a DeepAR Estimator
Some estimators have specific, SageMaker constructors, but not all. Instead you can create a base Estimator and pass in the specific image (or container) that holds a specific model.

Next, we configure the container image to be used for the region that we are running in.

In [24]:
from sagemaker.amazon.amazon_estimator import get_image_uri

image_name = get_image_uri(boto3.Session().region_name, # get the region
                           'forecasting-deepar') # specify image
EXERCISE: Instantiate an Estimator
You can now define the estimator that will launch the training job. A generic Estimator will be defined by the usual constructor arguments and an image_name.

You can take a look at the estimator source code to view specifics.

In [25]:
from sagemaker.estimator import Estimator

# dir to save model artifacts
s3_output_path = "s3://{}/{}/output".format(bucket, prefix)

# instantiate a DeepAR estimator
estimator = Estimator(sagemaker_session=sagemaker_session,
                      image_name=image_name,
                      role=role,
                      train_instance_count=1,
                      train_instance_type='ml.c4.xlarge',
                      output_path=s3_output_path
                      )
Setting Hyperparameters
Next, we need to define some DeepAR hyperparameters that define the model size and training behavior. Values for the epochs, frequency, prediction length, and context length are required.

epochs: The maximum number of times to pass over the data when training.
time_freq: The granularity of the time series in the dataset ('D' for daily).
prediction_length: A string; the number of time steps (based off the unit of frequency) that the model is trained to predict.
context_length: The number of time points that the model gets to see before making a prediction.
Context Length
Typically, it is recommended that you start with a context_length=prediction_length. This is because a DeepAR model also receives "lagged" inputs from the target time series, which allow the model to capture long-term dependencies. For example, a daily time series can have yearly seasonality and DeepAR automatically includes a lag of one year. So, the context length can be shorter than a year, and the model will still be able to capture this seasonality.

The lag values that the model picks depend on the frequency of the time series. For example, lag values for daily frequency are the previous week, 2 weeks, 3 weeks, 4 weeks, and year. You can read more about this in the DeepAR "how it works" documentation.

Optional Hyperparameters
You can also configure optional hyperparameters to further tune your model. These include parameters like the number of layers in our RNN model, the number of cells per layer, the likelihood function, and the training options, such as batch size and learning rate.

For an exhaustive list of all the different DeepAR hyperparameters you can refer to the DeepAR hyperparameter documentation.

In [26]:
freq='D'
context_length=30 # same as prediction_length

hyperparameters = {
    "epochs": "50",
    "time_freq": freq,
    "prediction_length": str(prediction_length),
    "context_length": str(context_length),
    "num_cells": "50",
    "num_layers": "2",
    "mini_batch_size": "128",
    "learning_rate": "0.001",
    "early_stopping_patience": "10"
}
In [27]:
# set the hyperparams
estimator.set_hyperparameters(**hyperparameters)
Training Job
Now, we are ready to launch the training job! SageMaker will start an EC2 instance, download the data from S3, start training the model and save the trained model.

If you provide the test data channel, as we do in this example, DeepAR will also calculate accuracy metrics for the trained model on this test data set. This is done by predicting the last prediction_length points of each time series in the test set and comparing this to the actual value of the time series. The computed error metrics will be included as part of the log output.

The next cell may take a few minutes to complete, depending on data size, model complexity, and training options.

In [28]:
%%time
# train and test channels
data_channels = {
    "train": train_path,
    "test": test_path
}

# fit the estimator
estimator.fit(inputs=data_channels)
INFO:sagemaker:Creating training-job with name: forecasting-deepar-2019-03-14-21-43-56-844
2019-03-14 21:43:56 Starting - Starting the training job...
2019-03-14 21:44:00 Starting - Launching requested ML instances......
2019-03-14 21:45:02 Starting - Preparing the instances for training......
2019-03-14 21:46:27 Downloading - Downloading input data...
2019-03-14 21:46:47 Training - Training image download completed. Training in progress.
Arguments: train
[03/14/2019 21:46:49 INFO 139824253740864] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'num_dynamic_feat': u'auto', u'dropout_rate': u'0.10', u'mini_batch_size': u'128', u'test_quantiles': u'[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]', u'_tuning_objective_metric': u'', u'_num_gpus': u'auto', u'num_eval_samples': u'100', u'learning_rate': u'0.001', u'num_cells': u'40', u'num_layers': u'2', u'embedding_dimension': u'10', u'_kvstore': u'auto', u'_num_kv_servers': u'auto', u'cardinality': u'auto', u'likelihood': u'student-t', u'early_stopping_patience': u''}
[03/14/2019 21:46:49 INFO 139824253740864] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'learning_rate': u'0.001', u'num_cells': u'50', u'prediction_length': u'30', u'epochs': u'50', u'time_freq': u'D', u'context_length': u'30', u'num_layers': u'2', u'mini_batch_size': u'128', u'early_stopping_patience': u'10'}
[03/14/2019 21:46:49 INFO 139824253740864] Final configuration: {u'dropout_rate': u'0.10', u'test_quantiles': u'[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]', u'_tuning_objective_metric': u'', u'num_eval_samples': u'100', u'learning_rate': u'0.001', u'num_layers': u'2', u'epochs': u'50', u'embedding_dimension': u'10', u'num_cells': u'50', u'_num_kv_servers': u'auto', u'mini_batch_size': u'128', u'likelihood': u'student-t', u'num_dynamic_feat': u'auto', u'cardinality': u'auto', u'_num_gpus': u'auto', u'prediction_length': u'30', u'time_freq': u'D', u'context_length': u'30', u'_kvstore': u'auto', u'early_stopping_patience': u'10'}
Process 1 is a worker.
[03/14/2019 21:46:49 INFO 139824253740864] Detected entry point for worker worker
[03/14/2019 21:46:50 INFO 139824253740864] Using early stopping with patience 10
[03/14/2019 21:46:50 INFO 139824253740864] [cardinality=auto] `cat` field was NOT found in the file `/opt/ml/input/data/train/train.json` and will NOT be used for training.
[03/14/2019 21:46:50 INFO 139824253740864] [num_dynamic_feat=auto] `dynamic_feat` field was NOT found in the file `/opt/ml/input/data/train/train.json` and will NOT be used for training.
[03/14/2019 21:46:50 INFO 139824253740864] Training set statistics:
[03/14/2019 21:46:50 INFO 139824253740864] Real time series
[03/14/2019 21:46:50 INFO 139824253740864] number of time series: 3
[03/14/2019 21:46:50 INFO 139824253740864] number of observations: 1006
[03/14/2019 21:46:50 INFO 139824253740864] mean target length: 335
[03/14/2019 21:46:50 INFO 139824253740864] min/mean/max target: 0.173818051815/1.05969123006/2.79841804504
[03/14/2019 21:46:50 INFO 139824253740864] mean abs(target): 1.05969123006
[03/14/2019 21:46:50 INFO 139824253740864] contains missing values: no
[03/14/2019 21:46:50 INFO 139824253740864] Small number of time series. Doing 10 number of passes over dataset per epoch.
[03/14/2019 21:46:50 INFO 139824253740864] Test set statistics:
[03/14/2019 21:46:50 INFO 139824253740864] Real time series
[03/14/2019 21:46:50 INFO 139824253740864] number of time series: 3
[03/14/2019 21:46:50 INFO 139824253740864] number of observations: 1096
[03/14/2019 21:46:50 INFO 139824253740864] mean target length: 365
[03/14/2019 21:46:50 INFO 139824253740864] min/mean/max target: 0.173818051815/1.08920513626/2.79841804504
[03/14/2019 21:46:50 INFO 139824253740864] mean abs(target): 1.08920513626
[03/14/2019 21:46:50 INFO 139824253740864] contains missing values: no
[03/14/2019 21:46:50 INFO 139824253740864] nvidia-smi took: 0.0252320766449 secs to identify 0 gpus
[03/14/2019 21:46:50 INFO 139824253740864] Number of GPUs being used: 0
[03/14/2019 21:46:50 INFO 139824253740864] Create Store: local
#metrics {"Metrics": {"get_graph.time": {"count": 1, "max": 160.68291664123535, "sum": 160.68291664123535, "min": 160.68291664123535}}, "EndTime": 1552600010.246369, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600010.084526}

[03/14/2019 21:46:50 INFO 139824253740864] Number of GPUs being used: 0
#metrics {"Metrics": {"initialize.time": {"count": 1, "max": 445.24383544921875, "sum": 445.24383544921875, "min": 445.24383544921875}}, "EndTime": 1552600010.529933, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600010.24646}

[03/14/2019 21:46:51 INFO 139824253740864] Epoch[0] Batch[0] avg_epoch_loss=2.302306
[03/14/2019 21:46:52 INFO 139824253740864] Epoch[0] Batch[5] avg_epoch_loss=1.403442
[03/14/2019 21:46:52 INFO 139824253740864] Epoch[0] Batch [5]#011Speed: 697.61 samples/sec#011loss=1.403442
[03/14/2019 21:46:52 INFO 139824253740864] processed a total of 1273 examples
#metrics {"Metrics": {"epochs": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "update.time": {"count": 1, "max": 2188.8790130615234, "sum": 2188.8790130615234, "min": 2188.8790130615234}}, "EndTime": 1552600012.719012, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600010.530032}

[03/14/2019 21:46:52 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=581.536132228 records/second
[03/14/2019 21:46:52 INFO 139824253740864] #progress_metric: host=algo-1, completed 2 % of epochs
[03/14/2019 21:46:52 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:46:52 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_c7020643-3eba-4e63-ad8c-3e9f1ceb51cf-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 26.62491798400879, "sum": 26.62491798400879, "min": 26.62491798400879}}, "EndTime": 1552600012.746186, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600012.719117}

[03/14/2019 21:46:53 INFO 139824253740864] Epoch[1] Batch[0] avg_epoch_loss=0.869504
[03/14/2019 21:46:54 INFO 139824253740864] Epoch[1] Batch[5] avg_epoch_loss=0.682243
[03/14/2019 21:46:54 INFO 139824253740864] Epoch[1] Batch [5]#011Speed: 602.43 samples/sec#011loss=0.682243
[03/14/2019 21:46:55 INFO 139824253740864] Epoch[1] Batch[10] avg_epoch_loss=0.626349
[03/14/2019 21:46:55 INFO 139824253740864] Epoch[1] Batch [10]#011Speed: 579.90 samples/sec#011loss=0.559276
[03/14/2019 21:46:55 INFO 139824253740864] processed a total of 1322 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2517.4388885498047, "sum": 2517.4388885498047, "min": 2517.4388885498047}}, "EndTime": 1552600015.263811, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600012.746291}

[03/14/2019 21:46:55 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=525.109128476 records/second
[03/14/2019 21:46:55 INFO 139824253740864] #progress_metric: host=algo-1, completed 4 % of epochs
[03/14/2019 21:46:55 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:46:55 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_774d06ca-3ce9-494b-8154-692bcd9436e3-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 32.75609016418457, "sum": 32.75609016418457, "min": 32.75609016418457}}, "EndTime": 1552600015.297069, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600015.263901}

[03/14/2019 21:46:55 INFO 139824253740864] Epoch[2] Batch[0] avg_epoch_loss=0.379552
[03/14/2019 21:46:56 INFO 139824253740864] Epoch[2] Batch[5] avg_epoch_loss=0.399700
[03/14/2019 21:46:56 INFO 139824253740864] Epoch[2] Batch [5]#011Speed: 713.90 samples/sec#011loss=0.399700
[03/14/2019 21:46:57 INFO 139824253740864] Epoch[2] Batch[10] avg_epoch_loss=0.387428
[03/14/2019 21:46:57 INFO 139824253740864] Epoch[2] Batch [10]#011Speed: 722.15 samples/sec#011loss=0.372701
[03/14/2019 21:46:57 INFO 139824253740864] processed a total of 1324 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2154.0908813476562, "sum": 2154.0908813476562, "min": 2154.0908813476562}}, "EndTime": 1552600017.451325, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600015.297157}

[03/14/2019 21:46:57 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=614.605605035 records/second
[03/14/2019 21:46:57 INFO 139824253740864] #progress_metric: host=algo-1, completed 6 % of epochs
[03/14/2019 21:46:57 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:46:57 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_22adec77-63ee-4869-9f33-b8c095d0f5c8-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 25.303125381469727, "sum": 25.303125381469727, "min": 25.303125381469727}}, "EndTime": 1552600017.477178, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600017.451416}

[03/14/2019 21:46:57 INFO 139824253740864] Epoch[3] Batch[0] avg_epoch_loss=0.329344
[03/14/2019 21:46:58 INFO 139824253740864] Epoch[3] Batch[5] avg_epoch_loss=0.320621
[03/14/2019 21:46:58 INFO 139824253740864] Epoch[3] Batch [5]#011Speed: 714.26 samples/sec#011loss=0.320621
[03/14/2019 21:46:59 INFO 139824253740864] processed a total of 1277 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1959.0270519256592, "sum": 1959.0270519256592, "min": 1959.0270519256592}}, "EndTime": 1552600019.436353, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600017.477256}

[03/14/2019 21:46:59 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=651.807311996 records/second
[03/14/2019 21:46:59 INFO 139824253740864] #progress_metric: host=algo-1, completed 8 % of epochs
[03/14/2019 21:46:59 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:46:59 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_169a6f15-6687-419f-83b3-2b85c2ae49ac-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 37.15085983276367, "sum": 37.15085983276367, "min": 37.15085983276367}}, "EndTime": 1552600019.474248, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600019.436449}

[03/14/2019 21:46:59 INFO 139824253740864] Epoch[4] Batch[0] avg_epoch_loss=0.265406
[03/14/2019 21:47:00 INFO 139824253740864] Epoch[4] Batch[5] avg_epoch_loss=0.260644
[03/14/2019 21:47:00 INFO 139824253740864] Epoch[4] Batch [5]#011Speed: 732.20 samples/sec#011loss=0.260644
[03/14/2019 21:47:01 INFO 139824253740864] Epoch[4] Batch[10] avg_epoch_loss=0.265656
[03/14/2019 21:47:01 INFO 139824253740864] Epoch[4] Batch [10]#011Speed: 725.47 samples/sec#011loss=0.271671
[03/14/2019 21:47:01 INFO 139824253740864] processed a total of 1299 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2163.55299949646, "sum": 2163.55299949646, "min": 2163.55299949646}}, "EndTime": 1552600021.637963, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600019.474335}

[03/14/2019 21:47:01 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=600.364963304 records/second
[03/14/2019 21:47:01 INFO 139824253740864] #progress_metric: host=algo-1, completed 10 % of epochs
[03/14/2019 21:47:01 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:01 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_01919238-a727-4831-b295-b458079913ba-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 31.22711181640625, "sum": 31.22711181640625, "min": 31.22711181640625}}, "EndTime": 1552600021.669655, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600021.638051}

[03/14/2019 21:47:02 INFO 139824253740864] Epoch[5] Batch[0] avg_epoch_loss=0.276591
[03/14/2019 21:47:02 INFO 139824253740864] Epoch[5] Batch[5] avg_epoch_loss=0.250042
[03/14/2019 21:47:02 INFO 139824253740864] Epoch[5] Batch [5]#011Speed: 729.89 samples/sec#011loss=0.250042
[03/14/2019 21:47:03 INFO 139824253740864] Epoch[5] Batch[10] avg_epoch_loss=0.233184
[03/14/2019 21:47:03 INFO 139824253740864] Epoch[5] Batch [10]#011Speed: 705.38 samples/sec#011loss=0.212954
[03/14/2019 21:47:03 INFO 139824253740864] processed a total of 1304 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2147.8641033172607, "sum": 2147.8641033172607, "min": 2147.8641033172607}}, "EndTime": 1552600023.817682, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600021.669743}

[03/14/2019 21:47:03 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=607.077495064 records/second
[03/14/2019 21:47:03 INFO 139824253740864] #progress_metric: host=algo-1, completed 12 % of epochs
[03/14/2019 21:47:03 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:03 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_cae87c06-fa92-48b6-89b2-856f3f9b2033-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 23.42391014099121, "sum": 23.42391014099121, "min": 23.42391014099121}}, "EndTime": 1552600023.84164, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600023.81777}

[03/14/2019 21:47:04 INFO 139824253740864] Epoch[6] Batch[0] avg_epoch_loss=0.197989
[03/14/2019 21:47:05 INFO 139824253740864] Epoch[6] Batch[5] avg_epoch_loss=0.216784
[03/14/2019 21:47:05 INFO 139824253740864] Epoch[6] Batch [5]#011Speed: 722.28 samples/sec#011loss=0.216784
[03/14/2019 21:47:05 INFO 139824253740864] processed a total of 1273 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1946.660041809082, "sum": 1946.660041809082, "min": 1946.660041809082}}, "EndTime": 1552600025.788445, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600023.84171}

[03/14/2019 21:47:05 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=653.8948524 records/second
[03/14/2019 21:47:05 INFO 139824253740864] #progress_metric: host=algo-1, completed 14 % of epochs
[03/14/2019 21:47:05 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:05 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_d1605b61-e40e-41c4-a4e6-6e515eeb0d45-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 23.60701560974121, "sum": 23.60701560974121, "min": 23.60701560974121}}, "EndTime": 1552600025.812576, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600025.788533}

[03/14/2019 21:47:06 INFO 139824253740864] Epoch[7] Batch[0] avg_epoch_loss=0.224201
[03/14/2019 21:47:07 INFO 139824253740864] Epoch[7] Batch[5] avg_epoch_loss=0.192646
[03/14/2019 21:47:07 INFO 139824253740864] Epoch[7] Batch [5]#011Speed: 731.57 samples/sec#011loss=0.192646
[03/14/2019 21:47:07 INFO 139824253740864] Epoch[7] Batch[10] avg_epoch_loss=0.185994
[03/14/2019 21:47:07 INFO 139824253740864] Epoch[7] Batch [10]#011Speed: 732.62 samples/sec#011loss=0.178013
[03/14/2019 21:47:07 INFO 139824253740864] processed a total of 1288 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2092.9009914398193, "sum": 2092.9009914398193, "min": 2092.9009914398193}}, "EndTime": 1552600027.905606, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600025.812636}

[03/14/2019 21:47:07 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=615.375521597 records/second
[03/14/2019 21:47:07 INFO 139824253740864] #progress_metric: host=algo-1, completed 16 % of epochs
[03/14/2019 21:47:07 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:07 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_460e86be-5ed7-4a19-898a-3bdbd2b80dbc-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 37.74595260620117, "sum": 37.74595260620117, "min": 37.74595260620117}}, "EndTime": 1552600027.943828, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600027.905693}

[03/14/2019 21:47:08 INFO 139824253740864] Epoch[8] Batch[0] avg_epoch_loss=0.161827
[03/14/2019 21:47:09 INFO 139824253740864] Epoch[8] Batch[5] avg_epoch_loss=0.134607
[03/14/2019 21:47:09 INFO 139824253740864] Epoch[8] Batch [5]#011Speed: 722.06 samples/sec#011loss=0.134607
[03/14/2019 21:47:10 INFO 139824253740864] Epoch[8] Batch[10] avg_epoch_loss=0.114896
[03/14/2019 21:47:10 INFO 139824253740864] Epoch[8] Batch [10]#011Speed: 725.67 samples/sec#011loss=0.091244
[03/14/2019 21:47:10 INFO 139824253740864] processed a total of 1294 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2205.5931091308594, "sum": 2205.5931091308594, "min": 2205.5931091308594}}, "EndTime": 1552600030.149585, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600027.943911}

[03/14/2019 21:47:10 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=586.652783829 records/second
[03/14/2019 21:47:10 INFO 139824253740864] #progress_metric: host=algo-1, completed 18 % of epochs
[03/14/2019 21:47:10 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:10 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_acf0ed81-79f3-4674-83d6-eb09f1550fbd-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 25.712013244628906, "sum": 25.712013244628906, "min": 25.712013244628906}}, "EndTime": 1552600030.175841, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600030.149679}

[03/14/2019 21:47:10 INFO 139824253740864] Epoch[9] Batch[0] avg_epoch_loss=0.141092
[03/14/2019 21:47:11 INFO 139824253740864] Epoch[9] Batch[5] avg_epoch_loss=0.131452
[03/14/2019 21:47:11 INFO 139824253740864] Epoch[9] Batch [5]#011Speed: 731.18 samples/sec#011loss=0.131452
[03/14/2019 21:47:12 INFO 139824253740864] processed a total of 1278 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1912.3728275299072, "sum": 1912.3728275299072, "min": 1912.3728275299072}}, "EndTime": 1552600032.088405, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600030.175957}

[03/14/2019 21:47:12 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=668.232828316 records/second
[03/14/2019 21:47:12 INFO 139824253740864] #progress_metric: host=algo-1, completed 20 % of epochs
[03/14/2019 21:47:12 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:12 INFO 139824253740864] Epoch[10] Batch[0] avg_epoch_loss=0.121748
[03/14/2019 21:47:13 INFO 139824253740864] Epoch[10] Batch[5] avg_epoch_loss=0.100604
[03/14/2019 21:47:13 INFO 139824253740864] Epoch[10] Batch [5]#011Speed: 696.88 samples/sec#011loss=0.100604
[03/14/2019 21:47:14 INFO 139824253740864] Epoch[10] Batch[10] avg_epoch_loss=0.090317
[03/14/2019 21:47:14 INFO 139824253740864] Epoch[10] Batch [10]#011Speed: 680.53 samples/sec#011loss=0.077973
[03/14/2019 21:47:14 INFO 139824253740864] processed a total of 1341 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2270.977020263672, "sum": 2270.977020263672, "min": 2270.977020263672}}, "EndTime": 1552600034.35983, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600032.088494}

[03/14/2019 21:47:14 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=590.459908479 records/second
[03/14/2019 21:47:14 INFO 139824253740864] #progress_metric: host=algo-1, completed 22 % of epochs
[03/14/2019 21:47:14 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:14 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_28061504-afef-437b-8731-5cea6a94d5ad-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 24.081945419311523, "sum": 24.081945419311523, "min": 24.081945419311523}}, "EndTime": 1552600034.384492, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600034.359921}

[03/14/2019 21:47:14 INFO 139824253740864] Epoch[11] Batch[0] avg_epoch_loss=0.125573
[03/14/2019 21:47:15 INFO 139824253740864] Epoch[11] Batch[5] avg_epoch_loss=0.097897
[03/14/2019 21:47:15 INFO 139824253740864] Epoch[11] Batch [5]#011Speed: 726.05 samples/sec#011loss=0.097897
[03/14/2019 21:47:16 INFO 139824253740864] Epoch[11] Batch[10] avg_epoch_loss=0.075347
[03/14/2019 21:47:16 INFO 139824253740864] Epoch[11] Batch [10]#011Speed: 727.50 samples/sec#011loss=0.048287
[03/14/2019 21:47:16 INFO 139824253740864] processed a total of 1315 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2131.3858032226562, "sum": 2131.3858032226562, "min": 2131.3858032226562}}, "EndTime": 1552600036.516015, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600034.384557}

[03/14/2019 21:47:16 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=616.930355504 records/second
[03/14/2019 21:47:16 INFO 139824253740864] #progress_metric: host=algo-1, completed 24 % of epochs
[03/14/2019 21:47:16 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:16 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_4a01b0d6-5484-4ad9-a81f-c1836b8f65e4-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 23.519039154052734, "sum": 23.519039154052734, "min": 23.519039154052734}}, "EndTime": 1552600036.540057, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600036.516104}

[03/14/2019 21:47:16 INFO 139824253740864] Epoch[12] Batch[0] avg_epoch_loss=0.047724
[03/14/2019 21:47:17 INFO 139824253740864] Epoch[12] Batch[5] avg_epoch_loss=0.090157
[03/14/2019 21:47:17 INFO 139824253740864] Epoch[12] Batch [5]#011Speed: 712.58 samples/sec#011loss=0.090157
[03/14/2019 21:47:18 INFO 139824253740864] processed a total of 1194 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1955.0070762634277, "sum": 1955.0070762634277, "min": 1955.0070762634277}}, "EndTime": 1552600038.495214, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600036.540118}

[03/14/2019 21:47:18 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=610.69672804 records/second
[03/14/2019 21:47:18 INFO 139824253740864] #progress_metric: host=algo-1, completed 26 % of epochs
[03/14/2019 21:47:18 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:18 INFO 139824253740864] Epoch[13] Batch[0] avg_epoch_loss=0.044313
[03/14/2019 21:47:19 INFO 139824253740864] Epoch[13] Batch[5] avg_epoch_loss=0.053000
[03/14/2019 21:47:19 INFO 139824253740864] Epoch[13] Batch [5]#011Speed: 714.21 samples/sec#011loss=0.053000
[03/14/2019 21:47:20 INFO 139824253740864] Epoch[13] Batch[10] avg_epoch_loss=0.038229
[03/14/2019 21:47:20 INFO 139824253740864] Epoch[13] Batch [10]#011Speed: 724.95 samples/sec#011loss=0.020503
[03/14/2019 21:47:20 INFO 139824253740864] processed a total of 1302 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2139.7829055786133, "sum": 2139.7829055786133, "min": 2139.7829055786133}}, "EndTime": 1552600040.635477, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600038.495307}

[03/14/2019 21:47:20 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=608.436266029 records/second
[03/14/2019 21:47:20 INFO 139824253740864] #progress_metric: host=algo-1, completed 28 % of epochs
[03/14/2019 21:47:20 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:20 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_fee716aa-b540-421d-aa24-349bfa8836f2-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 35.68696975708008, "sum": 35.68696975708008, "min": 35.68696975708008}}, "EndTime": 1552600040.67162, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600040.635564}

[03/14/2019 21:47:21 INFO 139824253740864] Epoch[14] Batch[0] avg_epoch_loss=0.072260
[03/14/2019 21:47:21 INFO 139824253740864] Epoch[14] Batch[5] avg_epoch_loss=0.037887
[03/14/2019 21:47:21 INFO 139824253740864] Epoch[14] Batch [5]#011Speed: 726.80 samples/sec#011loss=0.037887
[03/14/2019 21:47:22 INFO 139824253740864] processed a total of 1258 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1958.909034729004, "sum": 1958.909034729004, "min": 1958.909034729004}}, "EndTime": 1552600042.63067, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600040.671693}

[03/14/2019 21:47:22 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=642.153860833 records/second
[03/14/2019 21:47:22 INFO 139824253740864] #progress_metric: host=algo-1, completed 30 % of epochs
[03/14/2019 21:47:22 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:22 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_4dfccdd8-7830-4bc4-a362-7dec8c724c21-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 23.54598045349121, "sum": 23.54598045349121, "min": 23.54598045349121}}, "EndTime": 1552600042.654707, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600042.630752}

[03/14/2019 21:47:22 INFO 139824253740864] Epoch[15] Batch[0] avg_epoch_loss=0.080502
[03/14/2019 21:47:23 INFO 139824253740864] Epoch[15] Batch[5] avg_epoch_loss=0.044198
[03/14/2019 21:47:23 INFO 139824253740864] Epoch[15] Batch [5]#011Speed: 634.70 samples/sec#011loss=0.044198
[03/14/2019 21:47:24 INFO 139824253740864] processed a total of 1244 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2051.819086074829, "sum": 2051.819086074829, "min": 2051.819086074829}}, "EndTime": 1552600044.706658, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600042.654769}

[03/14/2019 21:47:24 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=606.254074807 records/second
[03/14/2019 21:47:24 INFO 139824253740864] #progress_metric: host=algo-1, completed 32 % of epochs
[03/14/2019 21:47:24 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:24 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_a488bf35-1c57-4a3e-be56-b41735ea2425-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 30.45511245727539, "sum": 30.45511245727539, "min": 30.45511245727539}}, "EndTime": 1552600044.737567, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600044.706743}

[03/14/2019 21:47:25 INFO 139824253740864] Epoch[16] Batch[0] avg_epoch_loss=0.053670
[03/14/2019 21:47:26 INFO 139824253740864] Epoch[16] Batch[5] avg_epoch_loss=0.038048
[03/14/2019 21:47:26 INFO 139824253740864] Epoch[16] Batch [5]#011Speed: 694.06 samples/sec#011loss=0.038048
[03/14/2019 21:47:26 INFO 139824253740864] processed a total of 1258 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1979.335069656372, "sum": 1979.335069656372, "min": 1979.335069656372}}, "EndTime": 1552600046.71705, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600044.73764}

[03/14/2019 21:47:26 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=635.525559249 records/second
[03/14/2019 21:47:26 INFO 139824253740864] #progress_metric: host=algo-1, completed 34 % of epochs
[03/14/2019 21:47:26 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:26 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_5d23ff3f-692f-4034-83a0-c28d5e2ecaef-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 25.405168533325195, "sum": 25.405168533325195, "min": 25.405168533325195}}, "EndTime": 1552600046.74298, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600046.717136}

[03/14/2019 21:47:27 INFO 139824253740864] Epoch[17] Batch[0] avg_epoch_loss=0.008913
[03/14/2019 21:47:27 INFO 139824253740864] Epoch[17] Batch[5] avg_epoch_loss=0.018935
[03/14/2019 21:47:27 INFO 139824253740864] Epoch[17] Batch [5]#011Speed: 709.59 samples/sec#011loss=0.018935
[03/14/2019 21:47:28 INFO 139824253740864] Epoch[17] Batch[10] avg_epoch_loss=0.015404
[03/14/2019 21:47:28 INFO 139824253740864] Epoch[17] Batch [10]#011Speed: 702.23 samples/sec#011loss=0.011167
[03/14/2019 21:47:28 INFO 139824253740864] processed a total of 1295 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2149.3399143218994, "sum": 2149.3399143218994, "min": 2149.3399143218994}}, "EndTime": 1552600048.89247, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600046.743054}

[03/14/2019 21:47:28 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=602.475476152 records/second
[03/14/2019 21:47:28 INFO 139824253740864] #progress_metric: host=algo-1, completed 36 % of epochs
[03/14/2019 21:47:28 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:28 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_0e999d3c-da95-4c94-bf06-535eb031ab88-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 23.684024810791016, "sum": 23.684024810791016, "min": 23.684024810791016}}, "EndTime": 1552600048.916647, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600048.892552}

[03/14/2019 21:47:29 INFO 139824253740864] Epoch[18] Batch[0] avg_epoch_loss=0.026352
[03/14/2019 21:47:30 INFO 139824253740864] Epoch[18] Batch[5] avg_epoch_loss=0.012523
[03/14/2019 21:47:30 INFO 139824253740864] Epoch[18] Batch [5]#011Speed: 703.88 samples/sec#011loss=0.012523
[03/14/2019 21:47:31 INFO 139824253740864] Epoch[18] Batch[10] avg_epoch_loss=-0.000198
[03/14/2019 21:47:31 INFO 139824253740864] Epoch[18] Batch [10]#011Speed: 728.52 samples/sec#011loss=-0.015464
[03/14/2019 21:47:31 INFO 139824253740864] processed a total of 1305 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2114.8791313171387, "sum": 2114.8791313171387, "min": 2114.8791313171387}}, "EndTime": 1552600051.031645, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600048.916706}

[03/14/2019 21:47:31 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=617.015153013 records/second
[03/14/2019 21:47:31 INFO 139824253740864] #progress_metric: host=algo-1, completed 38 % of epochs
[03/14/2019 21:47:31 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:31 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_7516855c-6e99-4487-8109-d999ca36e4c9-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 25.538206100463867, "sum": 25.538206100463867, "min": 25.538206100463867}}, "EndTime": 1552600051.057825, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600051.031743}

[03/14/2019 21:47:31 INFO 139824253740864] Epoch[19] Batch[0] avg_epoch_loss=0.042218
[03/14/2019 21:47:32 INFO 139824253740864] Epoch[19] Batch[5] avg_epoch_loss=0.015061
[03/14/2019 21:47:32 INFO 139824253740864] Epoch[19] Batch [5]#011Speed: 702.47 samples/sec#011loss=0.015061
[03/14/2019 21:47:32 INFO 139824253740864] processed a total of 1200 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1927.7119636535645, "sum": 1927.7119636535645, "min": 1927.7119636535645}}, "EndTime": 1552600052.985674, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600051.057896}

[03/14/2019 21:47:32 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=622.457296351 records/second
[03/14/2019 21:47:32 INFO 139824253740864] #progress_metric: host=algo-1, completed 40 % of epochs
[03/14/2019 21:47:32 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:33 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_623894fd-21e1-4a0d-b193-18bfac091878-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 28.3510684967041, "sum": 28.3510684967041, "min": 28.3510684967041}}, "EndTime": 1552600053.01451, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600052.985762}

[03/14/2019 21:47:33 INFO 139824253740864] Epoch[20] Batch[0] avg_epoch_loss=-0.003564
[03/14/2019 21:47:34 INFO 139824253740864] Epoch[20] Batch[5] avg_epoch_loss=-0.011826
[03/14/2019 21:47:34 INFO 139824253740864] Epoch[20] Batch [5]#011Speed: 697.79 samples/sec#011loss=-0.011826
[03/14/2019 21:47:35 INFO 139824253740864] Epoch[20] Batch[10] avg_epoch_loss=-0.031688
[03/14/2019 21:47:35 INFO 139824253740864] Epoch[20] Batch [10]#011Speed: 717.34 samples/sec#011loss=-0.055522
[03/14/2019 21:47:35 INFO 139824253740864] processed a total of 1281 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2135.190010070801, "sum": 2135.190010070801, "min": 2135.190010070801}}, "EndTime": 1552600055.149851, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600053.014588}

[03/14/2019 21:47:35 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=599.909830357 records/second
[03/14/2019 21:47:35 INFO 139824253740864] #progress_metric: host=algo-1, completed 42 % of epochs
[03/14/2019 21:47:35 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:35 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_15bf5951-07da-46b7-ae76-c7f5ab81ef37-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 37.35017776489258, "sum": 37.35017776489258, "min": 37.35017776489258}}, "EndTime": 1552600055.187666, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600055.149938}

[03/14/2019 21:47:35 INFO 139824253740864] Epoch[21] Batch[0] avg_epoch_loss=0.041688
[03/14/2019 21:47:36 INFO 139824253740864] Epoch[21] Batch[5] avg_epoch_loss=-0.005715
[03/14/2019 21:47:36 INFO 139824253740864] Epoch[21] Batch [5]#011Speed: 705.44 samples/sec#011loss=-0.005715
[03/14/2019 21:47:37 INFO 139824253740864] Epoch[21] Batch[10] avg_epoch_loss=-0.029992
[03/14/2019 21:47:37 INFO 139824253740864] Epoch[21] Batch [10]#011Speed: 731.79 samples/sec#011loss=-0.059124
[03/14/2019 21:47:37 INFO 139824253740864] processed a total of 1283 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2104.3410301208496, "sum": 2104.3410301208496, "min": 2104.3410301208496}}, "EndTime": 1552600057.292191, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600055.187744}

[03/14/2019 21:47:37 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=609.644806518 records/second
[03/14/2019 21:47:37 INFO 139824253740864] #progress_metric: host=algo-1, completed 44 % of epochs
[03/14/2019 21:47:37 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:37 INFO 139824253740864] Epoch[22] Batch[0] avg_epoch_loss=-0.019004
[03/14/2019 21:47:38 INFO 139824253740864] Epoch[22] Batch[5] avg_epoch_loss=-0.006339
[03/14/2019 21:47:38 INFO 139824253740864] Epoch[22] Batch [5]#011Speed: 716.07 samples/sec#011loss=-0.006339
[03/14/2019 21:47:39 INFO 139824253740864] Epoch[22] Batch[10] avg_epoch_loss=-0.041517
[03/14/2019 21:47:39 INFO 139824253740864] Epoch[22] Batch [10]#011Speed: 713.19 samples/sec#011loss=-0.083730
[03/14/2019 21:47:39 INFO 139824253740864] processed a total of 1297 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2142.3990726470947, "sum": 2142.3990726470947, "min": 2142.3990726470947}}, "EndTime": 1552600059.435007, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600057.292281}

[03/14/2019 21:47:39 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=605.359633776 records/second
[03/14/2019 21:47:39 INFO 139824253740864] #progress_metric: host=algo-1, completed 46 % of epochs
[03/14/2019 21:47:39 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:39 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_c9f147fd-1025-4b9a-87ad-e009fef2d81d-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 27.213096618652344, "sum": 27.213096618652344, "min": 27.213096618652344}}, "EndTime": 1552600059.46272, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600059.435093}

[03/14/2019 21:47:39 INFO 139824253740864] Epoch[23] Batch[0] avg_epoch_loss=-0.000329
[03/14/2019 21:47:40 INFO 139824253740864] Epoch[23] Batch[5] avg_epoch_loss=-0.030944
[03/14/2019 21:47:40 INFO 139824253740864] Epoch[23] Batch [5]#011Speed: 728.62 samples/sec#011loss=-0.030944
[03/14/2019 21:47:41 INFO 139824253740864] Epoch[23] Batch[10] avg_epoch_loss=-0.023259
[03/14/2019 21:47:41 INFO 139824253740864] Epoch[23] Batch [10]#011Speed: 724.39 samples/sec#011loss=-0.014037
[03/14/2019 21:47:41 INFO 139824253740864] processed a total of 1304 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2120.1560497283936, "sum": 2120.1560497283936, "min": 2120.1560497283936}}, "EndTime": 1552600061.583022, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600059.462794}

[03/14/2019 21:47:41 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=615.010752229 records/second
[03/14/2019 21:47:41 INFO 139824253740864] #progress_metric: host=algo-1, completed 48 % of epochs
[03/14/2019 21:47:41 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:41 INFO 139824253740864] Epoch[24] Batch[0] avg_epoch_loss=-0.006075
[03/14/2019 21:47:42 INFO 139824253740864] Epoch[24] Batch[5] avg_epoch_loss=-0.020238
[03/14/2019 21:47:42 INFO 139824253740864] Epoch[24] Batch [5]#011Speed: 729.29 samples/sec#011loss=-0.020238
[03/14/2019 21:47:43 INFO 139824253740864] processed a total of 1266 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1933.4380626678467, "sum": 1933.4380626678467, "min": 1933.4380626678467}}, "EndTime": 1552600063.516882, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600061.58311}

[03/14/2019 21:47:43 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=654.747386461 records/second
[03/14/2019 21:47:43 INFO 139824253740864] #progress_metric: host=algo-1, completed 50 % of epochs
[03/14/2019 21:47:43 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:43 INFO 139824253740864] Epoch[25] Batch[0] avg_epoch_loss=-0.017288
[03/14/2019 21:47:44 INFO 139824253740864] Epoch[25] Batch[5] avg_epoch_loss=-0.051539
[03/14/2019 21:47:44 INFO 139824253740864] Epoch[25] Batch [5]#011Speed: 696.22 samples/sec#011loss=-0.051539
[03/14/2019 21:47:45 INFO 139824253740864] processed a total of 1201 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1973.7861156463623, "sum": 1973.7861156463623, "min": 1973.7861156463623}}, "EndTime": 1552600065.491195, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600063.516971}

[03/14/2019 21:47:45 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=608.420130471 records/second
[03/14/2019 21:47:45 INFO 139824253740864] #progress_metric: host=algo-1, completed 52 % of epochs
[03/14/2019 21:47:45 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:45 INFO 139824253740864] Epoch[26] Batch[0] avg_epoch_loss=-0.077929
[03/14/2019 21:47:46 INFO 139824253740864] Epoch[26] Batch[5] avg_epoch_loss=-0.045151
[03/14/2019 21:47:46 INFO 139824253740864] Epoch[26] Batch [5]#011Speed: 711.30 samples/sec#011loss=-0.045151
[03/14/2019 21:47:47 INFO 139824253740864] Epoch[26] Batch[10] avg_epoch_loss=-0.016705
[03/14/2019 21:47:47 INFO 139824253740864] Epoch[26] Batch [10]#011Speed: 728.57 samples/sec#011loss=0.017430
[03/14/2019 21:47:47 INFO 139824253740864] processed a total of 1288 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2101.3739109039307, "sum": 2101.3739109039307, "min": 2101.3739109039307}}, "EndTime": 1552600067.593054, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600065.491282}

[03/14/2019 21:47:47 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=612.895540784 records/second
[03/14/2019 21:47:47 INFO 139824253740864] #progress_metric: host=algo-1, completed 54 % of epochs
[03/14/2019 21:47:47 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:47 INFO 139824253740864] Epoch[27] Batch[0] avg_epoch_loss=-0.033724
[03/14/2019 21:47:48 INFO 139824253740864] Epoch[27] Batch[5] avg_epoch_loss=-0.035065
[03/14/2019 21:47:48 INFO 139824253740864] Epoch[27] Batch [5]#011Speed: 695.00 samples/sec#011loss=-0.035065
[03/14/2019 21:47:49 INFO 139824253740864] processed a total of 1257 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2000.2851486206055, "sum": 2000.2851486206055, "min": 2000.2851486206055}}, "EndTime": 1552600069.593743, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600067.593136}

[03/14/2019 21:47:49 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=628.368687354 records/second
[03/14/2019 21:47:49 INFO 139824253740864] #progress_metric: host=algo-1, completed 56 % of epochs
[03/14/2019 21:47:49 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:49 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_62136251-c5bc-49b6-9a22-8c1871d6317e-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 28.232097625732422, "sum": 28.232097625732422, "min": 28.232097625732422}}, "EndTime": 1552600069.622484, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600069.593833}

[03/14/2019 21:47:49 INFO 139824253740864] Epoch[28] Batch[0] avg_epoch_loss=-0.043212
[03/14/2019 21:47:50 INFO 139824253740864] Epoch[28] Batch[5] avg_epoch_loss=-0.045693
[03/14/2019 21:47:50 INFO 139824253740864] Epoch[28] Batch [5]#011Speed: 694.92 samples/sec#011loss=-0.045693
[03/14/2019 21:47:51 INFO 139824253740864] Epoch[28] Batch[10] avg_epoch_loss=-0.030977
[03/14/2019 21:47:51 INFO 139824253740864] Epoch[28] Batch [10]#011Speed: 705.36 samples/sec#011loss=-0.013318
[03/14/2019 21:47:51 INFO 139824253740864] processed a total of 1314 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2173.321008682251, "sum": 2173.321008682251, "min": 2173.321008682251}}, "EndTime": 1552600071.796009, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600069.622592}

[03/14/2019 21:47:51 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=604.569637203 records/second
[03/14/2019 21:47:51 INFO 139824253740864] #progress_metric: host=algo-1, completed 58 % of epochs
[03/14/2019 21:47:51 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:52 INFO 139824253740864] Epoch[29] Batch[0] avg_epoch_loss=-0.082779
[03/14/2019 21:47:53 INFO 139824253740864] Epoch[29] Batch[5] avg_epoch_loss=-0.077948
[03/14/2019 21:47:53 INFO 139824253740864] Epoch[29] Batch [5]#011Speed: 706.06 samples/sec#011loss=-0.077948
[03/14/2019 21:47:53 INFO 139824253740864] Epoch[29] Batch[10] avg_epoch_loss=-0.077146
[03/14/2019 21:47:53 INFO 139824253740864] Epoch[29] Batch [10]#011Speed: 715.69 samples/sec#011loss=-0.076183
[03/14/2019 21:47:53 INFO 139824253740864] processed a total of 1356 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2123.608112335205, "sum": 2123.608112335205, "min": 2123.608112335205}}, "EndTime": 1552600073.920197, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600071.796098}

[03/14/2019 21:47:53 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=638.48721892 records/second
[03/14/2019 21:47:53 INFO 139824253740864] #progress_metric: host=algo-1, completed 60 % of epochs
[03/14/2019 21:47:53 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:53 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_ae756f20-9566-46f4-81c7-ac7cef00dffa-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 24.58500862121582, "sum": 24.58500862121582, "min": 24.58500862121582}}, "EndTime": 1552600073.945317, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600073.920291}

[03/14/2019 21:47:54 INFO 139824253740864] Epoch[30] Batch[0] avg_epoch_loss=-0.075710
[03/14/2019 21:47:55 INFO 139824253740864] Epoch[30] Batch[5] avg_epoch_loss=-0.059806
[03/14/2019 21:47:55 INFO 139824253740864] Epoch[30] Batch [5]#011Speed: 692.62 samples/sec#011loss=-0.059806
[03/14/2019 21:47:56 INFO 139824253740864] Epoch[30] Batch[10] avg_epoch_loss=-0.065062
[03/14/2019 21:47:56 INFO 139824253740864] Epoch[30] Batch [10]#011Speed: 717.46 samples/sec#011loss=-0.071369
[03/14/2019 21:47:56 INFO 139824253740864] processed a total of 1317 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2138.2100582122803, "sum": 2138.2100582122803, "min": 2138.2100582122803}}, "EndTime": 1552600076.083681, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600073.945399}

[03/14/2019 21:47:56 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=615.895430538 records/second
[03/14/2019 21:47:56 INFO 139824253740864] #progress_metric: host=algo-1, completed 62 % of epochs
[03/14/2019 21:47:56 INFO 139824253740864] loss did not improve
[03/14/2019 21:47:56 INFO 139824253740864] Epoch[31] Batch[0] avg_epoch_loss=-0.056232
[03/14/2019 21:47:57 INFO 139824253740864] Epoch[31] Batch[5] avg_epoch_loss=-0.067154
[03/14/2019 21:47:57 INFO 139824253740864] Epoch[31] Batch [5]#011Speed: 699.04 samples/sec#011loss=-0.067154
[03/14/2019 21:47:58 INFO 139824253740864] Epoch[31] Batch[10] avg_epoch_loss=-0.078518
[03/14/2019 21:47:58 INFO 139824253740864] Epoch[31] Batch [10]#011Speed: 724.28 samples/sec#011loss=-0.092155
[03/14/2019 21:47:58 INFO 139824253740864] processed a total of 1330 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2133.5220336914062, "sum": 2133.5220336914062, "min": 2133.5220336914062}}, "EndTime": 1552600078.217685, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600076.083773}

[03/14/2019 21:47:58 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=623.346137515 records/second
[03/14/2019 21:47:58 INFO 139824253740864] #progress_metric: host=algo-1, completed 64 % of epochs
[03/14/2019 21:47:58 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:47:58 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_0f996dfa-877c-45e6-9941-9cffc377ac8a-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 24.64604377746582, "sum": 24.64604377746582, "min": 24.64604377746582}}, "EndTime": 1552600078.242871, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600078.217765}

[03/14/2019 21:47:58 INFO 139824253740864] Epoch[32] Batch[0] avg_epoch_loss=-0.081447
[03/14/2019 21:47:59 INFO 139824253740864] Epoch[32] Batch[5] avg_epoch_loss=-0.082128
[03/14/2019 21:47:59 INFO 139824253740864] Epoch[32] Batch [5]#011Speed: 661.14 samples/sec#011loss=-0.082128
[03/14/2019 21:48:00 INFO 139824253740864] processed a total of 1272 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2001.8041133880615, "sum": 2001.8041133880615, "min": 2001.8041133880615}}, "EndTime": 1552600080.244846, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600078.24296}

[03/14/2019 21:48:00 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=635.38208496 records/second
[03/14/2019 21:48:00 INFO 139824253740864] #progress_metric: host=algo-1, completed 66 % of epochs
[03/14/2019 21:48:00 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:48:00 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_c4428484-fe9e-459a-995e-d659bebbe60a-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 23.881912231445312, "sum": 23.881912231445312, "min": 23.881912231445312}}, "EndTime": 1552600080.269254, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600080.244939}

[03/14/2019 21:48:00 INFO 139824253740864] Epoch[33] Batch[0] avg_epoch_loss=-0.041735
[03/14/2019 21:48:01 INFO 139824253740864] Epoch[33] Batch[5] avg_epoch_loss=-0.096359
[03/14/2019 21:48:01 INFO 139824253740864] Epoch[33] Batch [5]#011Speed: 703.36 samples/sec#011loss=-0.096359
[03/14/2019 21:48:02 INFO 139824253740864] Epoch[33] Batch[10] avg_epoch_loss=-0.126348
[03/14/2019 21:48:02 INFO 139824253740864] Epoch[33] Batch [10]#011Speed: 709.28 samples/sec#011loss=-0.162334
[03/14/2019 21:48:02 INFO 139824253740864] processed a total of 1308 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2128.19504737854, "sum": 2128.19504737854, "min": 2128.19504737854}}, "EndTime": 1552600082.397579, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600080.269321}

[03/14/2019 21:48:02 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=614.570345381 records/second
[03/14/2019 21:48:02 INFO 139824253740864] #progress_metric: host=algo-1, completed 68 % of epochs
[03/14/2019 21:48:02 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:48:02 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_a139a698-8fd3-47f1-85ca-dd4076625a28-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 24.560213088989258, "sum": 24.560213088989258, "min": 24.560213088989258}}, "EndTime": 1552600082.422648, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600082.397662}

[03/14/2019 21:48:02 INFO 139824253740864] Epoch[34] Batch[0] avg_epoch_loss=-0.086216
[03/14/2019 21:48:03 INFO 139824253740864] Epoch[34] Batch[5] avg_epoch_loss=-0.089683
[03/14/2019 21:48:03 INFO 139824253740864] Epoch[34] Batch [5]#011Speed: 675.70 samples/sec#011loss=-0.089683
[03/14/2019 21:48:04 INFO 139824253740864] Epoch[34] Batch[10] avg_epoch_loss=-0.052318
[03/14/2019 21:48:04 INFO 139824253740864] Epoch[34] Batch [10]#011Speed: 717.14 samples/sec#011loss=-0.007480
[03/14/2019 21:48:04 INFO 139824253740864] processed a total of 1299 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2158.8449478149414, "sum": 2158.8449478149414, "min": 2158.8449478149414}}, "EndTime": 1552600084.581613, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600082.422702}

[03/14/2019 21:48:04 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=601.67470324 records/second
[03/14/2019 21:48:04 INFO 139824253740864] #progress_metric: host=algo-1, completed 70 % of epochs
[03/14/2019 21:48:04 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:04 INFO 139824253740864] Epoch[35] Batch[0] avg_epoch_loss=-0.097148
[03/14/2019 21:48:05 INFO 139824253740864] Epoch[35] Batch[5] avg_epoch_loss=-0.099993
[03/14/2019 21:48:05 INFO 139824253740864] Epoch[35] Batch [5]#011Speed: 704.25 samples/sec#011loss=-0.099993
[03/14/2019 21:48:06 INFO 139824253740864] Epoch[35] Batch[10] avg_epoch_loss=-0.096781
[03/14/2019 21:48:06 INFO 139824253740864] Epoch[35] Batch [10]#011Speed: 722.82 samples/sec#011loss=-0.092926
[03/14/2019 21:48:06 INFO 139824253740864] processed a total of 1296 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2178.6649227142334, "sum": 2178.6649227142334, "min": 2178.6649227142334}}, "EndTime": 1552600086.760772, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600084.581701}

[03/14/2019 21:48:06 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=594.824241278 records/second
[03/14/2019 21:48:06 INFO 139824253740864] #progress_metric: host=algo-1, completed 72 % of epochs
[03/14/2019 21:48:06 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:07 INFO 139824253740864] Epoch[36] Batch[0] avg_epoch_loss=-0.100489
[03/14/2019 21:48:08 INFO 139824253740864] Epoch[36] Batch[5] avg_epoch_loss=-0.111643
[03/14/2019 21:48:08 INFO 139824253740864] Epoch[36] Batch [5]#011Speed: 718.92 samples/sec#011loss=-0.111643
[03/14/2019 21:48:08 INFO 139824253740864] processed a total of 1243 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1976.6669273376465, "sum": 1976.6669273376465, "min": 1976.6669273376465}}, "EndTime": 1552600088.737856, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600086.760859}

[03/14/2019 21:48:08 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=628.794325037 records/second
[03/14/2019 21:48:08 INFO 139824253740864] #progress_metric: host=algo-1, completed 74 % of epochs
[03/14/2019 21:48:08 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:09 INFO 139824253740864] Epoch[37] Batch[0] avg_epoch_loss=-0.051346
[03/14/2019 21:48:10 INFO 139824253740864] Epoch[37] Batch[5] avg_epoch_loss=-0.122393
[03/14/2019 21:48:10 INFO 139824253740864] Epoch[37] Batch [5]#011Speed: 693.23 samples/sec#011loss=-0.122393
[03/14/2019 21:48:10 INFO 139824253740864] Epoch[37] Batch[10] avg_epoch_loss=-0.123440
[03/14/2019 21:48:10 INFO 139824253740864] Epoch[37] Batch [10]#011Speed: 720.38 samples/sec#011loss=-0.124696
[03/14/2019 21:48:10 INFO 139824253740864] processed a total of 1302 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2201.6260623931885, "sum": 2201.6260623931885, "min": 2201.6260623931885}}, "EndTime": 1552600090.940005, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600088.737945}

[03/14/2019 21:48:10 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=591.344835192 records/second
[03/14/2019 21:48:10 INFO 139824253740864] #progress_metric: host=algo-1, completed 76 % of epochs
[03/14/2019 21:48:10 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:11 INFO 139824253740864] Epoch[38] Batch[0] avg_epoch_loss=-0.133864
[03/14/2019 21:48:12 INFO 139824253740864] Epoch[38] Batch[5] avg_epoch_loss=-0.105957
[03/14/2019 21:48:12 INFO 139824253740864] Epoch[38] Batch [5]#011Speed: 707.88 samples/sec#011loss=-0.105957
[03/14/2019 21:48:12 INFO 139824253740864] processed a total of 1217 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1922.9209423065186, "sum": 1922.9209423065186, "min": 1922.9209423065186}}, "EndTime": 1552600092.863439, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600090.940093}

[03/14/2019 21:48:12 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=632.844961 records/second
[03/14/2019 21:48:12 INFO 139824253740864] #progress_metric: host=algo-1, completed 78 % of epochs
[03/14/2019 21:48:12 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:13 INFO 139824253740864] Epoch[39] Batch[0] avg_epoch_loss=-0.137368
[03/14/2019 21:48:14 INFO 139824253740864] Epoch[39] Batch[5] avg_epoch_loss=-0.127098
[03/14/2019 21:48:14 INFO 139824253740864] Epoch[39] Batch [5]#011Speed: 699.52 samples/sec#011loss=-0.127098
[03/14/2019 21:48:14 INFO 139824253740864] processed a total of 1241 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1943.3529376983643, "sum": 1943.3529376983643, "min": 1943.3529376983643}}, "EndTime": 1552600094.807445, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600092.863534}

[03/14/2019 21:48:14 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=638.545285029 records/second
[03/14/2019 21:48:14 INFO 139824253740864] #progress_metric: host=algo-1, completed 80 % of epochs
[03/14/2019 21:48:14 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:48:14 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_80fb2ed7-095b-43ea-92ad-665fbe3bf212-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 23.754119873046875, "sum": 23.754119873046875, "min": 23.754119873046875}}, "EndTime": 1552600094.831874, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600094.807527}

[03/14/2019 21:48:15 INFO 139824253740864] Epoch[40] Batch[0] avg_epoch_loss=-0.098755
[03/14/2019 21:48:16 INFO 139824253740864] Epoch[40] Batch[5] avg_epoch_loss=-0.121262
[03/14/2019 21:48:16 INFO 139824253740864] Epoch[40] Batch [5]#011Speed: 703.03 samples/sec#011loss=-0.121262
[03/14/2019 21:48:16 INFO 139824253740864] Epoch[40] Batch[10] avg_epoch_loss=-0.088156
[03/14/2019 21:48:16 INFO 139824253740864] Epoch[40] Batch [10]#011Speed: 730.23 samples/sec#011loss=-0.048428
[03/14/2019 21:48:16 INFO 139824253740864] processed a total of 1301 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2106.1511039733887, "sum": 2106.1511039733887, "min": 2106.1511039733887}}, "EndTime": 1552600096.93817, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600094.831945}

[03/14/2019 21:48:16 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=617.674049309 records/second
[03/14/2019 21:48:16 INFO 139824253740864] #progress_metric: host=algo-1, completed 82 % of epochs
[03/14/2019 21:48:16 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:17 INFO 139824253740864] Epoch[41] Batch[0] avg_epoch_loss=-0.063164
[03/14/2019 21:48:18 INFO 139824253740864] Epoch[41] Batch[5] avg_epoch_loss=-0.112251
[03/14/2019 21:48:18 INFO 139824253740864] Epoch[41] Batch [5]#011Speed: 678.75 samples/sec#011loss=-0.112251
[03/14/2019 21:48:19 INFO 139824253740864] Epoch[41] Batch[10] avg_epoch_loss=-0.119862
[03/14/2019 21:48:19 INFO 139824253740864] Epoch[41] Batch [10]#011Speed: 726.76 samples/sec#011loss=-0.128994
[03/14/2019 21:48:19 INFO 139824253740864] processed a total of 1286 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2159.701108932495, "sum": 2159.701108932495, "min": 2159.701108932495}}, "EndTime": 1552600099.098374, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600096.938262}

[03/14/2019 21:48:19 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=595.384147469 records/second
[03/14/2019 21:48:19 INFO 139824253740864] #progress_metric: host=algo-1, completed 84 % of epochs
[03/14/2019 21:48:19 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:19 INFO 139824253740864] Epoch[42] Batch[0] avg_epoch_loss=-0.161080
[03/14/2019 21:48:20 INFO 139824253740864] Epoch[42] Batch[5] avg_epoch_loss=-0.152821
[03/14/2019 21:48:20 INFO 139824253740864] Epoch[42] Batch [5]#011Speed: 702.57 samples/sec#011loss=-0.152821
[03/14/2019 21:48:21 INFO 139824253740864] Epoch[42] Batch[10] avg_epoch_loss=-0.161127
[03/14/2019 21:48:21 INFO 139824253740864] Epoch[42] Batch [10]#011Speed: 722.95 samples/sec#011loss=-0.171094
[03/14/2019 21:48:21 INFO 139824253740864] processed a total of 1285 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2123.480796813965, "sum": 2123.480796813965, "min": 2123.480796813965}}, "EndTime": 1552600101.222635, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600099.098561}

[03/14/2019 21:48:21 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=605.100801284 records/second
[03/14/2019 21:48:21 INFO 139824253740864] #progress_metric: host=algo-1, completed 86 % of epochs
[03/14/2019 21:48:21 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:48:21 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_00def608-c0b8-495e-b292-f676ef37ec43-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 26.33810043334961, "sum": 26.33810043334961, "min": 26.33810043334961}}, "EndTime": 1552600101.249555, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600101.22272}

[03/14/2019 21:48:21 INFO 139824253740864] Epoch[43] Batch[0] avg_epoch_loss=-0.051589
[03/14/2019 21:48:22 INFO 139824253740864] Epoch[43] Batch[5] avg_epoch_loss=-0.093030
[03/14/2019 21:48:22 INFO 139824253740864] Epoch[43] Batch [5]#011Speed: 685.89 samples/sec#011loss=-0.093030
[03/14/2019 21:48:23 INFO 139824253740864] processed a total of 1232 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1981.6930294036865, "sum": 1981.6930294036865, "min": 1981.6930294036865}}, "EndTime": 1552600103.231407, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600101.249638}

[03/14/2019 21:48:23 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=621.651370708 records/second
[03/14/2019 21:48:23 INFO 139824253740864] #progress_metric: host=algo-1, completed 88 % of epochs
[03/14/2019 21:48:23 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:23 INFO 139824253740864] Epoch[44] Batch[0] avg_epoch_loss=-0.083385
[03/14/2019 21:48:24 INFO 139824253740864] Epoch[44] Batch[5] avg_epoch_loss=-0.128221
[03/14/2019 21:48:24 INFO 139824253740864] Epoch[44] Batch [5]#011Speed: 690.18 samples/sec#011loss=-0.128221
[03/14/2019 21:48:25 INFO 139824253740864] Epoch[44] Batch[10] avg_epoch_loss=-0.146216
[03/14/2019 21:48:25 INFO 139824253740864] Epoch[44] Batch [10]#011Speed: 702.83 samples/sec#011loss=-0.167811
[03/14/2019 21:48:25 INFO 139824253740864] processed a total of 1339 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2157.428026199341, "sum": 2157.428026199341, "min": 2157.428026199341}}, "EndTime": 1552600105.389313, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600103.231491}

[03/14/2019 21:48:25 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=620.606444649 records/second
[03/14/2019 21:48:25 INFO 139824253740864] #progress_metric: host=algo-1, completed 90 % of epochs
[03/14/2019 21:48:25 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:25 INFO 139824253740864] Epoch[45] Batch[0] avg_epoch_loss=-0.218158
[03/14/2019 21:48:26 INFO 139824253740864] Epoch[45] Batch[5] avg_epoch_loss=-0.159964
[03/14/2019 21:48:26 INFO 139824253740864] Epoch[45] Batch [5]#011Speed: 696.50 samples/sec#011loss=-0.159964
[03/14/2019 21:48:27 INFO 139824253740864] Epoch[45] Batch[10] avg_epoch_loss=-0.154195
[03/14/2019 21:48:27 INFO 139824253740864] Epoch[45] Batch [10]#011Speed: 719.54 samples/sec#011loss=-0.147273
[03/14/2019 21:48:27 INFO 139824253740864] processed a total of 1303 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2127.4938583374023, "sum": 2127.4938583374023, "min": 2127.4938583374023}}, "EndTime": 1552600107.517304, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600105.389407}

[03/14/2019 21:48:27 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=612.417072802 records/second
[03/14/2019 21:48:27 INFO 139824253740864] #progress_metric: host=algo-1, completed 92 % of epochs
[03/14/2019 21:48:27 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:27 INFO 139824253740864] Epoch[46] Batch[0] avg_epoch_loss=-0.192583
[03/14/2019 21:48:28 INFO 139824253740864] Epoch[46] Batch[5] avg_epoch_loss=-0.142080
[03/14/2019 21:48:28 INFO 139824253740864] Epoch[46] Batch [5]#011Speed: 703.81 samples/sec#011loss=-0.142080
[03/14/2019 21:48:29 INFO 139824253740864] processed a total of 1271 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1945.924997329712, "sum": 1945.924997329712, "min": 1945.924997329712}}, "EndTime": 1552600109.463731, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600107.5174}

[03/14/2019 21:48:29 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=653.112196958 records/second
[03/14/2019 21:48:29 INFO 139824253740864] #progress_metric: host=algo-1, completed 94 % of epochs
[03/14/2019 21:48:29 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:29 INFO 139824253740864] Epoch[47] Batch[0] avg_epoch_loss=-0.148459
[03/14/2019 21:48:30 INFO 139824253740864] Epoch[47] Batch[5] avg_epoch_loss=-0.161876
[03/14/2019 21:48:30 INFO 139824253740864] Epoch[47] Batch [5]#011Speed: 697.36 samples/sec#011loss=-0.161876
[03/14/2019 21:48:31 INFO 139824253740864] Epoch[47] Batch[10] avg_epoch_loss=-0.166682
[03/14/2019 21:48:31 INFO 139824253740864] Epoch[47] Batch [10]#011Speed: 712.80 samples/sec#011loss=-0.172450
[03/14/2019 21:48:31 INFO 139824253740864] processed a total of 1329 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2191.425085067749, "sum": 2191.425085067749, "min": 2191.425085067749}}, "EndTime": 1552600111.655757, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600109.463825}

[03/14/2019 21:48:31 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=606.416810477 records/second
[03/14/2019 21:48:31 INFO 139824253740864] #progress_metric: host=algo-1, completed 96 % of epochs
[03/14/2019 21:48:31 INFO 139824253740864] best epoch loss so far
[03/14/2019 21:48:31 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/state_19e75895-4cf8-4754-b601-f64a0a84f1d8-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 24.802207946777344, "sum": 24.802207946777344, "min": 24.802207946777344}}, "EndTime": 1552600111.681189, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600111.655848}

[03/14/2019 21:48:31 INFO 139824253740864] Epoch[48] Batch[0] avg_epoch_loss=-0.155690
[03/14/2019 21:48:32 INFO 139824253740864] Epoch[48] Batch[5] avg_epoch_loss=-0.155972
[03/14/2019 21:48:32 INFO 139824253740864] Epoch[48] Batch [5]#011Speed: 708.55 samples/sec#011loss=-0.155972
[03/14/2019 21:48:33 INFO 139824253740864] Epoch[48] Batch[10] avg_epoch_loss=-0.157168
[03/14/2019 21:48:33 INFO 139824253740864] Epoch[48] Batch [10]#011Speed: 727.88 samples/sec#011loss=-0.158605
[03/14/2019 21:48:33 INFO 139824253740864] processed a total of 1287 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2097.728967666626, "sum": 2097.728967666626, "min": 2097.728967666626}}, "EndTime": 1552600113.779063, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600111.681265}

[03/14/2019 21:48:33 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=613.484652454 records/second
[03/14/2019 21:48:33 INFO 139824253740864] #progress_metric: host=algo-1, completed 98 % of epochs
[03/14/2019 21:48:33 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:34 INFO 139824253740864] Epoch[49] Batch[0] avg_epoch_loss=-0.133067
[03/14/2019 21:48:35 INFO 139824253740864] Epoch[49] Batch[5] avg_epoch_loss=-0.144526
[03/14/2019 21:48:35 INFO 139824253740864] Epoch[49] Batch [5]#011Speed: 692.83 samples/sec#011loss=-0.144526
[03/14/2019 21:48:35 INFO 139824253740864] Epoch[49] Batch[10] avg_epoch_loss=-0.158706
[03/14/2019 21:48:35 INFO 139824253740864] Epoch[49] Batch [10]#011Speed: 716.85 samples/sec#011loss=-0.175722
[03/14/2019 21:48:35 INFO 139824253740864] processed a total of 1330 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2205.440044403076, "sum": 2205.440044403076, "min": 2205.440044403076}}, "EndTime": 1552600115.984905, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600113.779143}

[03/14/2019 21:48:35 INFO 139824253740864] #throughput_metric: host=algo-1, train throughput=603.019051488 records/second
[03/14/2019 21:48:35 INFO 139824253740864] #progress_metric: host=algo-1, completed 100 % of epochs
[03/14/2019 21:48:35 INFO 139824253740864] loss did not improve
[03/14/2019 21:48:35 INFO 139824253740864] Final loss: -0.166682457382 (occurred at epoch 47)
[03/14/2019 21:48:35 INFO 139824253740864] #quality_metric: host=algo-1, train final_loss <loss>=-0.166682457382
[03/14/2019 21:48:35 INFO 139824253740864] Worker algo-1 finished training.
[03/14/2019 21:48:35 WARNING 139824253740864] wait_for_all_workers will not sync workers since the kv store is not running distributed
[03/14/2019 21:48:35 INFO 139824253740864] All workers finished. Serializing model for prediction.
#metrics {"Metrics": {"get_graph.time": {"count": 1, "max": 198.6548900604248, "sum": 198.6548900604248, "min": 198.6548900604248}}, "EndTime": 1552600116.184355, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600115.984994}

[03/14/2019 21:48:36 INFO 139824253740864] Number of GPUs being used: 0
#metrics {"Metrics": {"finalize.time": {"count": 1, "max": 277.9650688171387, "sum": 277.9650688171387, "min": 277.9650688171387}}, "EndTime": 1552600116.26362, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600116.184427}

[03/14/2019 21:48:36 INFO 139824253740864] Serializing to /opt/ml/model/model_algo-1
[03/14/2019 21:48:36 INFO 139824253740864] Saved checkpoint to "/opt/ml/model/model_algo-1-0000.params"
#metrics {"Metrics": {"model.serialize.time": {"count": 1, "max": 20.006179809570312, "sum": 20.006179809570312, "min": 20.006179809570312}}, "EndTime": 1552600116.28374, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600116.263691}

[03/14/2019 21:48:36 INFO 139824253740864] Successfully serialized the model for prediction.
[03/14/2019 21:48:36 INFO 139824253740864] Evaluating model accuracy on testset using 100 samples
#metrics {"Metrics": {"model.bind.time": {"count": 1, "max": 0.03600120544433594, "sum": 0.03600120544433594, "min": 0.03600120544433594}}, "EndTime": 1552600116.284636, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600116.283797}

#metrics {"Metrics": {"model.score.time": {"count": 1, "max": 4311.997890472412, "sum": 4311.997890472412, "min": 4311.997890472412}}, "EndTime": 1552600120.596606, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600116.284684}

[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, RMSE): 0.347935478716
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, mean_wQuantileLoss): 0.142695
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.1]): 0.0733666
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.2]): 0.120334
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.3]): 0.152267
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.4]): 0.173388
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.5]): 0.182891
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.6]): 0.180538
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.7]): 0.167578
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.8]): 0.140406
[03/14/2019 21:48:40 INFO 139824253740864] #test_score (algo-1, wQuantileLoss[0.9]): 0.0934876
[03/14/2019 21:48:40 INFO 139824253740864] #quality_metric: host=algo-1, test RMSE <loss>=0.347935478716
[03/14/2019 21:48:40 INFO 139824253740864] #quality_metric: host=algo-1, test mean_wQuantileLoss <loss>=0.142695143819
#metrics {"Metrics": {"totaltime": {"count": 1, "max": 110749.34315681458, "sum": 110749.34315681458, "min": 110749.34315681458}, "setuptime": {"count": 1, "max": 9.984016418457031, "sum": 9.984016418457031, "min": 9.984016418457031}}, "EndTime": 1552600120.620557, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1552600120.596683}


2019-03-14 21:48:51 Uploading - Uploading generated training model
2019-03-14 21:48:51 Completed - Training job completed
Billable seconds: 144
CPU times: user 652 ms, sys: 40.4 ms, total: 693 ms
Wall time: 5min 12s
Deploy and Create a Predictor
Now that we have trained a model, we can use it to perform predictions by deploying it to a predictor endpoint.

Remember to delete the endpoint at the end of this notebook. A cell at the very bottom of this notebook will be provided, but it is always good to keep, front-of-mind.

In [29]:
%%time

# create a predictor
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    content_type="application/json" # specify that it will accept/produce JSON
)
INFO:sagemaker:Creating model with name: forecasting-deepar-2019-03-14-21-49-09-832
INFO:sagemaker:Creating endpoint with name forecasting-deepar-2019-03-14-21-43-56-844
---------------------------------------------------------------------------------------------------!CPU times: user 498 ms, sys: 22.2 ms, total: 520 ms
Wall time: 8min 21s
Generating Predictions
According to the inference format for DeepAR, the predictor expects to see input data in a JSON format, with the following keys:

instances: A list of JSON-formatted time series that should be forecast by the model.
configuration (optional): A dictionary of configuration information for the type of response desired by the request.
Within configuration the following keys can be configured:

num_samples: An integer specifying the number of samples that the model generates when making a probabilistic prediction.
output_types: A list specifying the type of response. We'll ask for quantiles, which look at the list of num_samples generated by the model, and generate quantile estimates for each time point based on these values.
quantiles: A list that specified which quantiles estimates are generated and returned in the response.
Below is an example of what a JSON query to a DeepAR model endpoint might look like.

{
 "instances": [
  { "start": "2009-11-01 00:00:00", "target": [4.0, 10.0, 50.0, 100.0, 113.0] },
  { "start": "1999-01-30", "target": [2.0, 1.0] }
 ],
 "configuration": {
  "num_samples": 50,
  "output_types": ["quantiles"],
  "quantiles": ["0.5", "0.9"]
 }
}
JSON Prediction Request
The code below accepts a list of time series as input and some configuration parameters. It then formats that series into a JSON instance and converts the input into an appropriately formatted JSON_input.

In [30]:
def json_predictor_input(input_ts, num_samples=50, quantiles=['0.1', '0.5', '0.9']):
    '''Accepts a list of input time series and produces a formatted input.
       :input_ts: An list of input time series.
       :num_samples: Number of samples to calculate metrics with.
       :quantiles: A list of quantiles to return in the predicted output.
       :return: The JSON-formatted input.
       '''
    # request data is made of JSON objects (instances)
    # and an output configuration that details the type of data/quantiles we want
    
    instances = []
    for k in range(len(input_ts)):
        # get JSON objects for input time series
        instances.append(series_to_json_obj(input_ts[k]))

    # specify the output quantiles and samples
    configuration = {"num_samples": num_samples, 
                     "output_types": ["quantiles"], 
                     "quantiles": quantiles}

    request_data = {"instances": instances, 
                    "configuration": configuration}

    json_request = json.dumps(request_data).encode('utf-8')
    
    return json_request
Get a Prediction
We can then use this function to get a prediction for a formatted time series!

In the next cell, I'm getting an input time series and known target, and passing the formatted input into the predictor endpoint to get a resultant prediction.

In [31]:
# get all input and target (test) time series
input_ts = time_series_training
target_ts = time_series

# get formatted input time series
json_input_ts = json_predictor_input(input_ts)

# get the prediction from the predictor
json_prediction = predictor.predict(json_input_ts)

#print(json_prediction)
Decoding Predictions
The predictor returns JSON-formatted prediction, and so we need to extract the predictions and quantile data that we want for visualizing the result. The function below, reads in a JSON-formatted prediction and produces a list of predictions in each quantile.

In [32]:
# helper function to decode JSON prediction
def decode_prediction(prediction, encoding='utf-8'):
    '''Accepts a JSON prediction and returns a list of prediction data.
    '''
    prediction_data = json.loads(prediction.decode(encoding))
    prediction_list = []
    for k in range(len(prediction_data['predictions'])):
        prediction_list.append(pd.DataFrame(data=prediction_data['predictions'][k]['quantiles']))
    return prediction_list
In [33]:
# get quantiles/predictions
prediction_list = decode_prediction(json_prediction)

# should get a list of 30 predictions 
# with corresponding quantile values
print(prediction_list[0])
         0.1       0.5       0.9
0   1.021641  1.421538  1.833977
1   1.071230  1.289308  1.571075
2   1.355773  1.592556  1.843872
3   1.131860  1.347549  1.530122
4   0.979925  1.247859  1.528625
5   1.221715  1.574440  1.866085
6   1.365926  1.701478  2.029046
7   1.289420  1.638611  1.903208
8   1.115153  1.320217  1.559577
9   1.174012  1.514868  1.788680
10  1.062032  1.439456  1.685213
11  1.002945  1.371620  1.670595
12  1.179278  1.535510  1.945749
13  1.031620  1.525921  1.950528
14  1.230493  1.507716  1.971212
15  0.929128  1.270095  1.679752
16  0.971372  1.425707  1.775358
17  0.874614  1.296613  1.896483
18  0.705493  1.233398  1.775203
19  0.909571  1.357466  1.983484
20  0.838604  1.475715  2.098177
21  0.635874  1.423548  1.985564
22  0.736272  1.134092  1.615218
23  0.601333  1.166549  1.677038
24  0.723368  1.076509  1.559197
25  0.502419  0.954523  1.373580
26  0.687602  1.235270  1.820810
27  0.692387  1.348745  2.049263
28  0.722391  1.146618  2.171784
29  0.458222  1.087659  1.615523
Display the Results!
The quantile data will give us all we need to see the results of our prediction.

Quantiles 0.1 and 0.9 represent higher and lower bounds for the predicted values.
Quantile 0.5 represents the median of all sample predictions.
In [34]:
# display the prediction median against the actual data
def display_quantiles(prediction_list, target_ts=None):
    # show predictions for all input ts
    for k in range(len(prediction_list)):
        plt.figure(figsize=(12,6))
        # get the target month of data
        if target_ts is not None:
            target = target_ts[k][-prediction_length:]
            plt.plot(range(len(target)), target, label='target')
        # get the quantile values at 10 and 90%
        p10 = prediction_list[k]['0.1']
        p90 = prediction_list[k]['0.9']
        # fill the 80% confidence interval
        plt.fill_between(p10.index, p10, p90, color='y', alpha=0.5, label='80% confidence interval')
        # plot the median prediction line
        prediction_list[k]['0.5'].plot(label='prediction median')
        plt.legend()
        plt.show()
In [35]:
# display predictions
display_quantiles(prediction_list, target_ts)



Predicting the Future
Recall that we did not give our model any data about 2010, but let's see if it can predict the energy consumption given no target, only a known start date!

EXERCISE: Format a request for a "future" prediction
Create a formatted input to send to the deployed predictor passing in my usual parameters for "configuration". The "instances" will, in this case, just be one instance, defined by the following:

start: The start time will be time stamp that you specify. To predict the first 30 days of 2010, start on Jan. 1st, '2010-01-01'.
target: The target will be an empty list because this year has no, complete associated time series; we specifically withheld that information from our model, for testing purposes.
{"start": start_time, "target": []} # empty target
In [36]:
# Starting my prediction at the beginning of 2010
start_date = '2010-01-01'
timestamp = '00:00:00'

# formatting start_date
start_time = start_date +' '+ timestamp

# formatting request_data
# this instance has an empty target!
request_data = {"instances": [{"start": start_time, "target": []}],
                "configuration": {"num_samples": 50,
                                  "output_types": ["quantiles"],
                                  "quantiles": ['0.1', '0.5', '0.9']}
                }

json_input = json.dumps(request_data).encode('utf-8')

print('Requesting prediction for '+start_time)
Requesting prediction for 2010-01-01 00:00:00
Then get and decode the prediction response, as usual.

In [37]:
# get prediction response
json_prediction = predictor.predict(json_input)

prediction_2010 = decode_prediction(json_prediction)
Finally, I'll compare the predictions to a known target sequence. This target will come from a time series for the 2010 data, which I'm creating below.

In [38]:
# create 2010 time series
ts_2010 = []

# get global consumption data
# index 1112 is where the 2010 data starts
data_2010 = mean_power_df.values[1112:]

index = pd.DatetimeIndex(start=start_date, periods=len(data_2010), freq='D')
ts_2010.append(pd.Series(data=data_2010, index=index))
In [39]:
# range of actual data to compare
start_idx=0 # days since Jan 1st 2010
end_idx=start_idx+prediction_length

# get target data
target_2010_ts = [ts_2010[0][start_idx:end_idx]]

# display predictions
display_quantiles(prediction_2010, target_2010_ts)

Delete the Endpoint
Try your code out on different time series. You may want to tweak your DeepAR hyperparameters and see if you can improve the performance of this predictor.

When you're done with evaluating the predictor (any predictor), make sure to delete the endpoint.

In [40]:
## TODO: delete the endpoint
predictor.delete_endpoint()
INFO:sagemaker:Deleting endpoint configuration with name: forecasting-deepar-2019-03-14-21-43-56-844
INFO:sagemaker:Deleting endpoint with name: forecasting-deepar-2019-03-14-21-43-56-844
Conclusion
Now you've seen one complex but far-reaching method for time series forecasting. You should have the skills you need to apply the DeepAR model to data that interests you!
