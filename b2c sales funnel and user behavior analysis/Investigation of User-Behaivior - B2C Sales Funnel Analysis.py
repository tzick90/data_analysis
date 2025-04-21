#!/usr/bin/env python
# coding: utf-8

# # Investigation of User-Behaivior / B2C Sales Funnel Analysis

# ## Project Description

# We work at at startup that sells food products. We need to investigate user behavior for the company's app.
# For this task, we are going to study the sales funnel and find out how users reach the purchase stage. how many users actually make it to this stage? <br>
# How mony get stuck at previous stages? <br>
# Which stages in particular? <br>
# For this, we need to look at the results of an A/A/B test.<br>
# <br>
# 
# Our goal is to investigate users behaivior.How they are behaving in each stage of the product. Also our designers would like to change the fonts for the entire app but managers are afraid the users might find the new design intimidating. So to make decision whether we should incorporate the new changes, we need to perform A/A/B test. Based on the result our manager will decide for the implementation of new changes.
# 
# 

#  

# --- 

#  

# ## Executive Summary

# 1. After loading the data, in the Data Preprocessing task, we found around 413 duplicated value which we dropped otherwise it will impact the analysis and also added two different columns for date and time.
# 2. After plotting histogram, We found some anomalies in the data. There is no enough data before 1-August and this is very less in % of total data approx 1.16% So for analysis we take data from 1-August till 07-Aug only.
# 3. After analyzing the event funnel, we can assume the sequence of the action are like: MainScreenAppear > OffersScreenAppear > CartScreenAppear > PaymentScreenSuccessful. Tutorial is optional as users does not require to visit the Tutorial. So we may not inlcude it as part of sequence. We can also ignore OffersScreenAppear as part of sequence as user can visit directly to Cartscreen or Paymentscreen without visiting the offerscreen.

#  

# ---

#  

# ## Table of Content
# <a id='toc'></a>

# #### [1. Open the data file and read the general information](#1)
# [1.1 Loading Libraries](#11)<br>
# [1.2 Loading dataset](#12)<br>
# [1.3 Understanding the data](#13)<br>
# 
# #### [2. Data pre processing](#2)<br>
# 
# #### [3. Studying and monitoring the data](#3)
# [3.1 How many events are in the logs?](#31)<br>
# [3.2 How many users are in the logs?](#32)<br>
# [3.3 What's the average number of events per user?](#33)<br>
# [3.4 What period of time does the data cover?](#34)<br>
# [3.5 At which moment is the data complete?](#35)<br>
# [3.6 Ensuring to include all users from all three experimental groups](#36)<br>
# 
# #### [4. Studying the event funnel](#4)
# [4.1 What events are in the logs and their frequency of occurrence?](#41)<br>
# [4.2 Number of users who performed each of these actions](#42)<br>
# [4.3 Order, in which the actions took place](#43)<br>
# [4.4 What period of time does the data cover?](#44)<br>
# [4.5 At which stage do we lose the most users?](#45)<br>
# [4.6 What share of users make the entire journey from their first event to payment?](#46)<br>
# 
# #### [5. Studying the results of the experiment](#5)<br>
# [5.1 How many users are there in each group?](#51)<br>
# [5.2 Statistically significant difference between samples 246 and 247](#52)<br>
# [5.3 Most popular event in each control group](#53)<br>
# [5.4 Most popular event for the group with altered fonds](#54)<br>
# [5.5 About the significance level for the tests](#55)<br>
# 
# #### [6. Conclusion ](#6)<br>

#  

#  

# ## 1. Open the data file and read the general information
# <a id='1'></a>
# <a id='1B'></a>

# [1.1 Loading Libraries](#11)<br>
# [1.2 Loading dataset](#12)<br>
# [1.2 Understanding the data](#13)<br>

#  

# ### 1.1 Loading Libraries
# <a id='11'></a>

# In[ ]:


get_ipython().system('pip install -q plotly')


# In[ ]:


get_ipython().system('pip install plotly -U')


# In[ ]:


get_ipython().system('pip install --upgrade -q seaborn')


# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from functools import reduce
from io import BytesIO
import requests
import os

import plotly.express as px
from plotly import graph_objects as go

import time
import datetime as dt

import math
from scipy import stats


import sidetable

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_colwidth', 400)


# Libraries loaded successfully.

#  

#  

#  

# ### 1.2 Loading dataset
# <a id='12'></a>

# In[3]:


try:
    food_data = pd.read_csv(r'/datasets/logs_exp_us.csv.')
    
except:
    dataset_id = '1QhusiA1NYyK0H0WESkH_j_x_II-sFPzU6t1ZX4m194g'
    food_data_sample_id_file = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv'.format(dataset_id)
    r1 = requests.get(food_data_sample_id_file)
    food_data = pd.read_csv(BytesIO(r1.content))


# Dataset loaded successfully.

#  

#  

#  

# ### 1.3 Understanding the data
# <a id='13'></a>

# In[4]:


# first look
display(food_data.head(),
        food_data.tail(),
        food_data.sample()
)


# It is always important to get a proper understanding of the data before starting the analyzation.
# It seems we need to convert the `EventTimestamp` into datetime format. <br>
# In addition, it would be wise to lower string the column head.

#  

#  

# In[5]:


# general infos and missing values
display(food_data.info(memory_usage= 'deep'))
print()
display(food_data.stb.missing(style= True))


# Gladfully, no missing values here. So we can proceed to check for duplicates.

#  

#  

# In[6]:


# checking for duplicates

print(f'Count of duplicate records: {food_data.duplicated().sum()}')
duplicate_data = food_data[food_data.duplicated()]
print(f'Percentage of duplicate records are: {round(len(duplicate_data)/len(food_data)*100,2)} %')


# We have a small amount of duplicated values, so we should get rid of them.

#  

#  

# In[7]:


# removing duplicated values
food_data.drop_duplicates(inplace=True)


# done.

# ---
# [back to: chapter start](#1B) | [back to: table of content](#toc)

#  

#  

#  

# ## 2. Data pre processing
# <a id='2'></a>
# <a id='2B'></a>

# In[8]:


# Renaming columns
food_data.rename(columns= {'EventName':'event_name',
                          'DeviceIDHash':'user_id',
                          'EventTimestamp':'event_datetime',
                          'ExpId':'exp_id'
                             }, inplace= True)
display(food_data)


# renaming the columns was successful.

#  

#  

# In[9]:


# Changing event_datetime to datetime values
food_data['event_datetime'] = pd.to_datetime(food_data['event_datetime'], unit='s')

# Dividing event_datetime into two seperated columns
food_data['event_date'] = food_data['event_datetime'].dt.date
food_data['event_time'] = food_data['event_datetime'].dt.time


food_data


# We converted the timestamp of the events into the correct format. Also, we split up the column into date(day) and date(time) columns for processing.

# ---
# [back to: chapter start](#2B) | [back to: table of content](#toc)

#  

#  

#  

# ## 3. Studying and monitoring the data
# <a id='3'></a>
# <a id='3B'></a>

# [3.1 How many events are in the logs?](#31)<br>
# [3.2 How many users are in the logs?](#32)<br>
# [3.3 What's the average number of events per user?](#33)<br>
# [3.4 What period of time does the data cover?](#34)<br>
# [3.5 At which moment is the data complete?](#35)<br>
# [3.6 Ensuring to include all users from all three experimental groups](#36)<br>

#  

#  

# ### 3.1 How many events are in the logs?
# <a id='31'></a>

# In[10]:


print(f'There are {food_data.shape[0]} events in the logs.')


# The dataset contains 243'713 events in the logs.

#  

#  

# ### 3.2 How many users are in the logs?
# <a id='32'></a>

# In[11]:


print(f'There are {food_data.user_id.nunique()} unique users in the logs.')


# The dataset contains 7551 unique users in the logs.

#  

#  

# ### 3.3 What's the average number of events per user?
# <a id='33'></a>

# In[12]:


data_user_event = food_data.groupby('user_id')['event_name'].agg('count').reset_index().sort_values(by=['user_id','event_name'])

data_user_event_ = food_data.groupby(['event_name']).agg({'user_id':['nunique','count']}).reset_index()
data_user_event_.columns = ['event_name','unique_user_count','total_visit']
data_user_event_['avg_visit_per_users'] = data_user_event_.total_visit/data_user_event_.unique_user_count


display(data_user_event_)
print()
print(f'Average number of events per user: {int(data_user_event.event_name.mean())}')


# On average, a user of the given dataset has 32 events.

#  

#  

# ### 3.4 What period of time does the data cover?
# <a id='34'></a>

# In[13]:


# plot histogram
sns.set(style="darkgrid")

fig_dima = (15, 6)
fig, axs = plt.subplots(figsize= fig_dima)
sns.histplot(data= food_data,
             x= "event_datetime",
             color= "slategray",
             kde= True,
             ax= axs ,
             label= 'Event Date')

axs.set_xlabel('Event Date', size= 16)
axs.set_ylabel('No. of records', size= 16)
axs.set_title('Histogram of Event Date and Time', size= 20)

axs.legend()

for item in axs.get_xticklabels():
    item.set_rotation(45)
fig.show()



min_event_date = food_data.event_datetime.min()
max_event_date = food_data.event_datetime.max()
print()
print(f'Minimum Date: {min_event_date}')
print(f'Maximum Date: {max_event_date}')


# 1. All the data are from the period between 2019-07-25 and 2019-08-07.
# 2. As we see from the above graph, before 1-August-2019 there is no enough data for the analysis. So we take data from 1-August till 07-Aug. <br>
# 
# There are very small number of events before 1-August. This might be due to various reasons: <br>
# 
# 1. There may be some technical issues on the app prior and it gets resolved in august which incresed the events.
# 2. Company expands/starts its services in other profitable region from august.
# 3. Comapny started its online booking/services from august.
# 4. It has introduced some new menu from august.
# 5. It has invested in advertising before which has started to pay off from august etc.

#  

#  

# ### 3.5 At which moment is the data complete?
# <a id='35'></a>

# In[14]:


non_included_data = food_data[food_data['event_datetime']<'2019-08-01']
food_data_final = food_data[food_data['event_datetime']>='2019-08-01']

print(f'% of data which will not be inlcuded if delete older data are: {round(non_included_data.shape[0]/food_data.shape[0]*100,2)}')



#Getting users list which we will not inlcude, if delete older data.
non_included_data_users = non_included_data[~non_included_data.user_id.isin(food_data_final.user_id)]

#display(non_included_data_users)
print()
print('Total no. of unique users which will not be included if delete older data are: {0}, which is almost {1:.2f}% of total unique users'
      .format(non_included_data_users.user_id.nunique(), 
              (non_included_data_users.user_id.nunique()/food_data.user_id.nunique()*100)
             ))


# 1. % of data which will not be inlcuded if delete older data are: 1.16%
# 2. Total no. of unique users which will not be included if delete older data are 17, which is almost 0.23% of total unique users
# 3. So now we can say from the above calculations that older data are very less in % of all data, so we can delete that and can continue with further analysis.

#  

#  

# ### 3.6 Ensuring to include all users from all three experimental groups
# <a id='36'></a>

# In[15]:


display(food_data_final.groupby('exp_id')['user_id'].nunique().reset_index())


# All the experiments contains enough data for analysis.

# ### Conclusions and Summary

# So far, we have loaded the data. Rename, added the new columns as per the convenient. Deleted the duplcate rows. Cheked all the data for any inappropriate data. We found some older data/events which are very less as compare to august data if we include this it can skew our data as well as can impact our analysis. So we decided to remove those outlier first and then will perform the further analysis.

# ---
# [back to: chapter start](#3B) | [back to: table of content](#toc)

#  

#  

#  

# ## 4. Studying the event funnel
# <a id='4'></a>
# <a id='4B'></a>

# [4.1 What events are in the logs and their frequency of occurrence?](#41)<br>
# [4.2 Number of users who performed each of these actions](#42)<br>
# [4.3 Order, in which the actions took place](#43)<br>
# [4.4 Event Funnel](#44)<br>
# [4.5 At which stage do we lose the most users?](#45)<br>
# [4.6 What share of users make the entire journey from their first event to payment?](#46)<br>

#  

#  

# ### 4.1 What events are in the logs and their frequency of occurrence?
# <a id='41'></a>

# In[16]:


data_event_freq = food_data.event_name.value_counts().rename_axis('event_name').reset_index(name= 'count').sort_values(by= 'count', ascending=False)
data_event_freq['%_to_total_data'] = round((data_event_freq['count']/len(food_data_final)*100),2)
display(data_event_freq)


# As we can see from the above table, 'MainScreenAppear' has highest frequency approx. half of total entries. Offerscreen, Cartscreen and Paymentscreen are approx. 19%, 17% and 14%. There is a less '%' of decrease in these three screen as compare to first screen. Tutorial has the least no. of visits i.e. only 0.43%.

#  

#  

# ### 4.2 Number of users who performed each of these actions
# <a id='42'></a>

# In[17]:


# users actions sorted by number of users which had performed these actions 
food_data_user_actions = food_data_final.groupby('event_name')['user_id'].nunique().rename_axis('event_name').reset_index(name= 'user_count').sort_values(by= 'user_count', ascending= False)


#print((data_foodLogs_final.user_id.nunique()))

# proportion of users
food_data_user_actions['in_%'] = round((food_data_user_actions.user_count/food_data_final.user_id.nunique())*100,2)
display(food_data_user_actions)


# So here we can say that most of users i.e. 98.47% of users vists the MainScreenAppear. After that 60.96% of them visit OffersScreenAppear, 49.56% CartScreenAppear and 46.97% visits PaymentScreenSuccessful. only 11.15% of them visits Tutorial.

#  

#  

# ### 4.3 Order, in which the actions took place
# <a id='43'></a>

# After looking at the above table, we can assume the sequence of the action like: 
# 
#  MainScreenAppear
#  >  OffersScreenAppear
#  >>  CartScreenAppear
#  >>>  PaymentScreenSuccessful. 
#  
#  Tutorial is optional as users does not require to visit the Tutorial. So we may not inlcude it as part of sequence.
# 
# There are also chances that without visiting Offers screen, user can directly go to the CartScreen or PaymentScreen but in all cases users must visit to the Main Screen and for making purchase Payment screen.

#  

#  

# ### 4.4 Event Funnel
# <a id='44'></a>

# In[18]:


user_actions_funnel = food_data_user_actions
user_actions_funnel['pct_change'] = user_actions_funnel['user_count'].pct_change()


# In[19]:



user_actions_funnel_group = []
food_data_final = food_data_final[food_data_final.event_name != 'Tutorial']
for i in food_data_final.exp_id.unique():
    df = food_data_final[food_data_final.exp_id == i]    .groupby(['event_name','exp_id'])['user_id']    .nunique().reset_index().sort_values(by= 'user_id', ascending= False)
   
    user_actions_funnel_group.append(df)
    
    

data_funnel_groups = pd.concat(user_actions_funnel_group)
data_funnel_groups = data_funnel_groups.rename(columns= {'event_name': 'Event Name',
                                                         'exp_id': 'EXP ID',
                                                        'user_id': 'User ID'})

# plotting the funnel

fig = px.funnel(data_funnel_groups,
                x= 'User ID',
                y= 'Event Name',
                color= 'EXP ID',
                title='Displaying users share in each event through Funnel chart')

fig.show()
display(data_funnel_groups)


# In[20]:


display(user_actions_funnel)


# As we can see from the data_funnel_shift table, there is a high % of decrease i.e. 38% in OffersScreen from MainScreen. From Offerscreen to CartScreen it is 18% and from CartScreen to Payment it is only 5%. In Tutorial also 76% decrease but it is not of much interest as this is not required screen to the users.

#  

#  

# ### 4.5 At which stage do we lose the most users?
# <a id='45'></a>

# We can say here that we lose most of users at OffersScreen i.e. approx. 38% after Tutorial screen but as Tutorial is not that much inportant to us so we need to concentrate on OffersScreenAppear.

#  

#  

# ### 4.6 What share of users make the entire journey from their first event to payment?
# <a id='46'></a>

# In[35]:


df_pvt_minTime =food_data_final[food_data_final.event_name != 'Tutorial'].pivot_table(index='user_id',
             columns='event_name',
             values='event_datetime',
             aggfunc='min')

#display(df_pvt_minTime)

df_pvt_minTime = df_pvt_minTime[['MainScreenAppear','PaymentScreenSuccessful']]
df_pvt_minTime = df_pvt_minTime-df_pvt_minTime.shift(+1,axis=1)
display(df_pvt_minTime)
print()
print('Share of users, who walked through the entire customer journey from Main Screen to Payment Screen: {}'      .format(str(round(len(df_pvt_minTime[df_pvt_minTime.PaymentScreenSuccessful.notnull()])/len(df_pvt_minTime)*100,3)) +' %'))
      


        


# Approx. 46% of users make the entire journey from MainScreen to payment screen.
# 
# We can increase the conversion of customer by limiting the loss of customers at each event like at Mainscreen we can display some good/promising offer to the users so that they tends towards next screen.

# ---
# [back to: chapter start](#4B) | [back to: table of content](#toc)

#  

#  

#  

# ## 5. Studying the results of the experiment
# <a id='5'></a>
# <a id='5B'></a>

# [5.1 How many users are there in each group?](#51)<br>
# [5.2 Statistically significant difference between samples 246 and 247](#52)<br>
# [5.3 Most popular event in each control group](#53)<br>
# [5.4 Most popular event for the group with altered fonds](#54)<br>
# [5.5 About the significance level for the tests](#55)<br>

#  

#  

# ### 5.1 How many users are there in each group?
# <a id='51'></a>

# In[22]:


user_group_participents = food_data_final.groupby('exp_id')['user_id'].nunique().reset_index(name='user_count')
user_group_participents['in_pct'] = round((user_group_participents.user_count/sum(user_group_participents.user_count)*100),2)
display(user_group_participents)


# Every group has almost same amount of users but group 246 has smallest amount (2484) as compare to other groups.

#  

#  

# ### 5.2 Statistically significant difference between samples 246 and 247
# <a id='52'></a>

# In[25]:


ssd = food_data_final.pivot_table(index= 'event_name',
                                  values= 'user_id',
                                  columns= 'exp_id',
                                  aggfunc= lambda x: x.nunique()).reset_index()




# Creating function to find statistical significance between groups for particular event.

def check_hypothesis(group1,group2, event, alpha=0.05):
    
    # let's start with successes, using 
    
    if (group1 == '246,247'):
        ssd['246_247_combined'] = (ssd[246] + ssd[247])
        successes1 = ssd[ssd.event_name == event]['246_247_combined'].iloc[0]
        
    else:
        successes1 = ssd[ssd.event_name == event][group1].iloc[0]
        
    successes2 = ssd[ssd.event_name == event][group2].iloc[0]
    
    # for trials we can go back to original df or used a pre-aggregated data
    if (group1 == '246,247'): # This condition works when we want to compare combined (246,247) group with 248.
        trials1 = food_data_final[food_data_final.exp_id.isin([246,247])]['user_id'].nunique()
    else:
        trials1 = food_data_final[food_data_final.exp_id == group1]['user_id'].nunique()
        
    trials2 = food_data_final[food_data_final.exp_id == group2]['user_id'].nunique()
    
    #proportion for success in the first group
    p1 = successes1/trials1

   #proportion for success in the second group
    p2 = successes2/trials2

    # proportion in a combined dataset
    p_combined = (successes1 + successes2) / (trials1 + trials2)

  
    difference = p1 - p2
    
    
    z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1/trials1 + 1/trials2))

  
    distr = stats.norm(0, 1) 


    p_value = (1 - distr.cdf(abs(z_value))) * 2

    print('p-value: ', p_value)

    if (p_value < alpha):
        print("Reject H0 for",event, 'and groups',group1,' and ' ,group2)
    else:
        print("Fail to Reject H0 for", event,'and groups',group1,' and ',group2)


# **H<sub>0</sub>**: There is **no** statistically significant difference between samples 246 and 247.<br>
# **H<sub>1</sub>**: There is **a** statistically significant difference between samples 246 and 247. <br>
# **$\alpha$** = 0.05 (*critical level of statistical significance*)

# In[26]:


for evt in food_data_final.event_name.unique():
    check_hypothesis(246,247, evt, alpha = 0.05)


# The p-value is greater than significance value which implies 'Fail to Reject H0'. i.e. There is no statistically significant difference between samples 246 and 247 for all events.

#  

#  

# ### 5.3 Most popular event in each control group
# <a id='53'></a>

# #### Select the most popular event. In each of the control groups, find the number of users who performed this action

# In[27]:


display(ssd)


# As we can see from the above dataframe, most popular event is MainScreenAppear. It also display number of users for each event and each group.

#  

# #### Check whether the difference between the groups is statistically significant. Repeat the procedure for all other events.
# 
# **H<sub>0</sub>**: There is **no** statistically significant difference between groups.<br>
# **H<sub>1</sub>**: There is **a** statistically significant difference between groups. <br>
# **$\alpha$** = 0.05 (*critical level of statistical significance*)

# In[28]:


check_hypothesis(246,247, 'MainScreenAppear', alpha= 0.05)


# In[29]:


for evt in food_data_final.event_name.unique():
    check_hypothesis(246,247, evt, alpha= 0.05)


# The p-value is greater than significance value (0.05) which implies 'Fail to Reject H0'. i.e. There is no statistically significant difference between both groups (246 and 247) for all events.

#  

#  

# ### 5.4 Most popular event for the group with altered fonds
# <a id='54'></a>
# 
# **H<sub>0</sub>**: There is **no** statistically significant difference between groups. <br>
# **H<sub>1</sub>**: There is **a** statistically significant difference between groups.<br>
# **$\alpha$** = 0.05 (*critical level of statistical significance*)

# In[30]:


for evt in food_data_final.event_name.unique():
    check_hypothesis(246,248, evt, alpha= 0.05)
    check_hypothesis(247,248, evt, alpha= 0.05)


# In[31]:


for evt in food_data_final.event_name.unique():
    check_hypothesis('246,247',248, evt, alpha= 0.05)


# The p-value is greater than significance value (0.05) which implies 'Fail to Reject H<sub>1</sub>. i.e. There is no statistically significant difference between groups. We checked various combinations between the control group and test group like
# 
# 1. 246 and 248
# 2. 247 and 248
# 3. 246 + 247 and 248. for all these combination with significance value 0.05, we found no statistically significant difference. thats means all groups are alomost same for all event.

#  

#  

# ### 5.5 About the significance level for the tests
# <a id='55'></a>

# #### What significance level have you set to test the statistical hypotheses mentioned above?

# **$\alpha$** = 0.05 (*critical level of statistical significance*)

#  

# #### Calculate how many statistical hypothesis tests you carried out.

# There are 5 events and 3 groups so as per this 15 statistical hypothesis tests needs to be carried out.

# #### With a statistical significance level of 0.1, one in 10 results could be false¶
# 
# **H<sub>0</sub>**: There is **no** statistically significant difference between groups. <br>
# **H<sub>1</sub>**: There is **a** statistically significant difference between groups.<br>
# **$\alpha$** = 0.01 (*critical level of statistical significance*)

#  

# #### What should the significance level be?

# In[33]:


# Setting the significance level using Bonferroni correction 

alpha = 0.05
n_hypothesis = 15
bonferroni_alpha = alpha / n_hypothesis  # three comparisons made
bonferroni_alpha = round(bonferroni_alpha,3)
print(bonferroni_alpha)


#  

# **H<sub>0</sub>**: There is **no** statistically significant difference between groups. <br>
# **H<sub>1</sub>**: There is **a** statistically significant difference between groups.<br>
# **$\alpha$** = 0.003 (*critical level of statistical significance - result benferroni_alpha*)

# In[36]:


#for i in pivot.event_name.unique():
 #   check_hypothesis(246,248, i, alpha= bonferroni_alpha)
  #  check_hypothesis(247,248, i, alpha= bonferroni_alpha)
   # check_hypothesis('246,247',248, i, alpha= bonferroni_alpha)


# The p-value is greater than significance value (bonferroni_alpha) which implies 'Fail to Reject H<sub>1</sub>. i.e. There is no statistically significant difference between groups.
# Here we use Bonferroni correction method for setting the alpha value as there are multiple test(15) which we need to run.
# But if we look at the pValue, This is already greater than existing alpha value(0.05) and now after applying Bonferroni correction it again reduced so find no use to apply this method. <br>
# 
# As per theory also by lower down the significance level we make more probability of type 2 error and In our case the above test already shows that there is no statistical significant difference between the groups and after decreasing the alpha value it will not change rather it will increase more probability of type 2 error.

# ---
# [back to: chapter start](#5B) | [back to: table of content](#toc)

#  

#  

#  

# ## 6. Conclusion
# <a id='6'></a>

# **Below are main points from our observation:**
# 
# 1. After loading the data, in the Data Preprocessing task, we found around 413 duplicated value which we dropped otherwise it will impact the analysis and also added two different columns for date and time.
# 2. After plotting histogram, We found some anomalies in the data. There is no enough data before 1-August and this is very less in % of total data approx 1.16% So for analysis we take data from 1-August till 07-Aug only.
# 3. After analyzing the event funnel, we can assume the sequence of the action are like: MainScreenAppear > OffersScreenAppear > CartScreenAppear > PaymentScreenSuccessful. Tutorial is optional as users does not require to visit the Tutorial. So we may not inlcude it as part of sequence. We can also ignore OffersScreenAppear as part of sequence as user can visit directly to Cartscreen or Paymentscreen without visiting the offerscreen.
# 4. There is a high % of decrease in user conversion i.e. 38% in OffersScreen from MainScreen. From Offerscreen to CartScreen it is 18% and from CartScreen to Payment it is only 5%. So we can say there is greater chance of conversion if user also visit to offerscreen from mainscreen so we can recommend our teams to add some most promising offers on the mainscreen to get more conversion from mainscreen.
# 5. Approx. 46% of users are reaching to payment screen which infact is a good number but still we can increase the conversion if can retain customer from Mainscreen to nextscreen.
# 6. In A/A/B testing, we checked various combination of control groups and test groups but found no statistically significant difference between groups. So we can say there is no statistically significant difference between the groups.At last, we can conclude that as there is no difference between the groups conversion after altering the fonts so we can reject the suggeston given by designer.

# ---
# [back to: table of content](#toc)
