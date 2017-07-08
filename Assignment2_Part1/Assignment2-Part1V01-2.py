
# coding: utf-8

# # EDA

# ## Import of the CSV file 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import boto
import boto.s3
from boto.s3.key import Key
from boto.s3.cors import CORSConfiguration
import glob
import boto3
import botocore
import csv
import re
import json
import os
import sys
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

print('import libraries ,packages')


# In[2]:

properties =pd.read_csv('properties_2016.csv',low_memory=False)
train =pd.read_csv('train_2016_v2.csv')
print('Step 1. Import of Data Done')


# ### Analysis on the Training Data 

# ### logerror outliers removal

# In[6]:

from bokeh.io import output_file, show
from bokeh.charts import Scatter
import bokeh.plotting as bk
from bokeh.plotting import figure 
bk.output_notebook()
plot = figure(plot_width=600,plot_height=200,tools ='pan,box_zoom')
s = pd.Series(range(train.shape[0]),name='index')
dataf = pd.DataFrame(s)
dataf["logerror"]= np.sort(train.logerror.values)
p = Scatter(dataf,x="index",y="logerror",xlabel="Row Number",ylabel="Value of Logerror ")

bk.show(p)


# ### Removing the outliers of Train csv Data 

# In[7]:

train = train[train.logerror <=3]
train = train[train.logerror >=-2.5]
print("Removing the outliers of Train csv Data")


# ### frequency of transactions each month

# In[8]:


#frequency of transactions each month
from bokeh.charts import Bar, output_notebook, show
import bokeh.charts as bc
import bokeh.plotting as bk
train['transactiondate']=pd.to_datetime(train['transactiondate'],format= '%Y-%m-%d')
q=train['transactiondate'].dt.month.value_counts()
df=pd.DataFrame(q)
p = Bar(df, 'index', values='transactiondate', title="Frequerncy of Transactions over the Months ",
                     xlabel="Month Number",ylabel="Frequency/Count",bar_width=0.3,color='wheat')
p.legend.visible= False
bk.show(p)



# ### We can easily see inthe monthof 11, 12 there is a drastic fall of the sales of home

# ## Analysis on Properties CSV data 

# ### Calculating number of null values present in the csv 

# In[9]:

properties.shape


# ### Getting the count of null values 

# In[10]:

properties.isnull().sum().reset_index()


# ### Calculating the Actual percentage of Null values present in each columns 

# In[11]:

df2=properties.isnull().sum().reset_index()
df2.columns=['column_name', 'missing_Percentage']
c=properties.parcelid.count()
for i,v in df2.missing_Percentage.iteritems():
    df2.missing_Percentage[i]=(v*100)/c

df2


# In[12]:

### Bar Graph representaion of the data
from bokeh.charts import Bar, output_notebook, show
import bokeh.charts as bc
p = figure(plot_width=2000,plot_height=600)
p = Bar(df2, 'column_name', values='missing_Percentage', title="Frequerncy of Transactions over the Months ",
                     xlabel="Month Number",ylabel="Frequency/Count",bar_width=0.3,color='cornflowerblue')
p.legend.visible= False
output_file("bar.html")

bc.show(p)



# ### Removing the Columns from the CSV which are not populated more than 60 % 

# In[13]:

for the_key, the_value in df2.missing_Percentage.iteritems():
    if the_value>=60:
#          del properties[the_key]
           del properties[str(df2.column_name[the_key])]


merged_data=pd.merge(properties,train,on='parcelid',how='left')


# In[14]:

merged_data.shape


# ### Removing the NAN vlaues as no the values for other columns are not present after 01 - OCT 2016

# In[15]:

merged_data=merged_data[(merged_data['transactiondate'] <='2016-10-01')]


# In[16]:

merged_data.info()


# ### So we have total 31 Columns , from which we need to Select our best variables to analyse 
# <p> We will review our most important column i.e assesed value of the Parcel 

# In[17]:

merged_data['structuretaxvaluedollarcnt'].describe()


# ### Assesed Value of Parcel is Positively Skewed Data 

# In[18]:

plt.figure(figsize=(20, 6))
sns.distplot(merged_data['structuretaxvaluedollarcnt'].dropna());
plt.show()


# ### For normally distributed data, the skewness should be about 0. A skewness value > 0 means that there is more weight in the left tail of the distribution.
# ### Krtosis: the sharpness of the peak of a frequency-distribution curve.

# In[19]:

#skewness and kurtosis
print("Skewness: %f" % merged_data['structuretaxvaluedollarcnt'].skew())
print("Kurtosis: %f" % merged_data['structuretaxvaluedollarcnt'].kurt())


# ### Normalizing the data for Latitude and Longitude 

# In[20]:

lo =np.array(merged_data['longitude']/1000000)
la=np.array(merged_data['latitude']/1000000)


# ### Plotting the positions of the parcel 

# In[21]:

plt.figure(figsize=(100,100))
sns.jointplot(x=la, y=lo, size=5,color='lightgreen')
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


# ### Plotting Assesed Value and Calulated Total Area of the Parcel 
# <p> There are outliers in our data and we have roughly direct relationship , although not true for most of the parcels </p>
# 

# In[22]:

#scatter plot

data = pd.concat([merged_data['structuretaxvaluedollarcnt'], merged_data['calculatedfinishedsquarefeet']/100], axis=1)
data.plot.scatter(x='structuretaxvaluedollarcnt', y='calculatedfinishedsquarefeet');


# ### Plot for Assesd Value and Land Tax Value of all parcels

# In[23]:

#scatter plot 

data = pd.concat([merged_data['structuretaxvaluedollarcnt'], merged_data['landtaxvaluedollarcnt']], axis=1)
data.plot.scatter(y='structuretaxvaluedollarcnt', x='landtaxvaluedollarcnt');


# ### Vizualization of Categorical DATA 
# <p>Plotting Building Type VS Assesed Value of the parcel </p>
# <p>Rating 1 (Worst) to 10(Supreme)</p>

# In[24]:

#box plot 
data = pd.concat([merged_data['structuretaxvaluedollarcnt'], merged_data['buildingqualitytypeid']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='buildingqualitytypeid', y="structuretaxvaluedollarcnt", data=data)
fig.axis();


# ### we can clearly see the number of selling high parcel value have been in rising,though half of data for each box plot is on average having same assessed Value 

# In[25]:

data = pd.concat([merged_data['structuretaxvaluedollarcnt'], merged_data['yearbuilt']], axis=1)
f, ax = plt.subplots(figsize=(20, 10))
fig = sns.boxplot(x='yearbuilt', y="structuretaxvaluedollarcnt", data=data)


# ### Finding Co-relation between all the available Columns 

# In[26]:

co=merged_data.corr()


# In[27]:

### Plot of Correlativity Matrix 


# In[28]:

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(co, vmax=.8, square=True);


# ### <p> Its clear fromthe Heat map that we may have some duplicate columns present tin the data as many of the columns shows Deep Red </p>
# <p> Lets Take top most Scored form the data Corelativitiy matrix </p>

# In[29]:

merged_data1=merged_data.dropna()


# In[30]:


k = 10 #number of variables for heatmap
cols = co.nlargest(k, 'structuretaxvaluedollarcnt')['structuretaxvaluedollarcnt'].index
cm = np.corrcoef(merged_data1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:




# ### We can easily find that the below Columns are same in the dataset 
# <p> finishedsquarefeet12              and calculatedfinishedsquarefeet     </p>
# <p> fullbathcnt                       , calculatedbathnbr     and      bathroomcnt                      </p> 
# 

# ### So we would remove those columns and make the Corelativity matrix again

# In[31]:

merged_data2=merged_data1.drop(['calculatedbathnbr','finishedsquarefeet12','fullbathcnt'],axis=1)


# In[32]:

co2=merged_data2.corr()
k = 10 #number of variables for heatmap
cols = co2.nlargest(k, 'structuretaxvaluedollarcnt')['structuretaxvaluedollarcnt'].index
cm = np.corrcoef(merged_data2[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[33]:

merged_data2.info()


# ### We would  vizualize all the important parameter thatcame up in the above matrix 
# <p> One interesting relation we can found out that propertylandusetypeid  is having -0.34 (negatively) realted with the BedRoom Count , So as the Bed Room Count Increases Land Usage Number decreases which makes sense as the id are scaled from the Best to Worst</p>

# In[34]:

sns.pairplot(merged_data2[['structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt','calculatedfinishedsquarefeet','bathroomcnt','bedroomcnt','propertylandusetypeid']])
plt.show()


# ### Removing features with low variance with respect to structuretaxvaluedollarcnt 
# <p> structuretaxvaluedollarcnt - The assessed value of the built structure on the parcel</p>
# <p>We are trying findout the most relevant Columns with respect to price of the assesed value of house </p>

# In[35]:

co["index1"]=co.index


# In[36]:

high_corr=[]
for i,row in co.iterrows():
    if (row.structuretaxvaluedollarcnt>-0.007 and row.structuretaxvaluedollarcnt<0.007)or (row.index1=='logerror' or row.index1=='assessmentyear'):
         print("")
    else:
        high_corr.append(row.index1)
high_corr.append('transactiondate')


# ### Moving the data to new Dataframe 

# In[37]:

imp_features=merged_data[high_corr]


# ### We have columns which are highly co related statiscally with the assessed value of each house 

# In[38]:

imp_features.columns


# In[39]:

high_corr


# ### By studying the values manually we found out that few columns have same value through the data , so its better to remove those columns from our analaysis  

# In[40]:

imp_features=imp_features.drop(['calculatedbathnbr','finishedsquarefeet12','fullbathcnt'],axis=1)


# ### After analysing we dropping few more columns whose Correlation and the data which are insignificant to out analysis 

# In[41]:

imp_features=imp_features.drop(['taxvaluedollarcnt','landtaxvaluedollarcnt'],axis=1)


# In[42]:

imp_features.columns


# ### Now we will try to fill the missing Data 

# In[43]:

imp_features.info()


# ## Total Number of Null value present 

# In[44]:

imp_features.parcelid.count()-imp_features.buildingqualitytypeid.count()


# ### We are filling the null values present in the buildingqualitytypeid with the average ID avlaible i.e 6  Range is from 1(best)- 12(worst) 

# In[45]:

imp_features.buildingqualitytypeid=imp_features.buildingqualitytypeid.fillna('6')


# ### We are removing the rest of the Rows whichare having Null values  as the there is very small percentage of Null values present for the rest of the Fields 

# In[46]:

total = imp_features.isnull().sum().sort_values(ascending=False)
percent = (imp_features.isnull().sum()/imp_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# In[47]:

imp_features=imp_features.dropna()


# ## Analysing Relationship between Total Area VS No of Bathroom present in the Parcel

# In[48]:

analysis1 = imp_features[["calculatedfinishedsquarefeet","bathroomcnt"]]


# ## Rescaling the data before plotting 

# In[49]:

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
np_scaled = min_max_scaler.fit_transform(analysis1)



# In[50]:

norm= pd.DataFrame(np_scaled)


# In[51]:

norm.columns=['calculatedfinishedsquarefeet','bathroomcnt']


# In[52]:

norm.head()


# In[53]:

import seaborn as sns
import matplotlib.pyplot as plt
sns.jointplot(y=norm['calculatedfinishedsquarefeet'] ,x=norm['bathroomcnt'])
plt.show() 


# ##  Removing the Outliers  Bivariate anlysis 

# In[54]:

#scatter plot 

data = pd.concat([imp_features['structuretaxvaluedollarcnt'], imp_features['taxamount']], axis=1)
data.plot.scatter(y='structuretaxvaluedollarcnt', x='taxamount');


# In[55]:

imp_features.sort_values(by = 'taxamount', ascending = False)[:3]


# In[56]:

imp_features = imp_features.drop(imp_features[imp_features['parcelid'] == 11648750].index)


# ### Univariate Analysis on structuretaxvaluedollarcnt column

# In[57]:

#standardizing data
from sklearn.preprocessing import StandardScaler
saleprice_scaled = StandardScaler().fit_transform(imp_features['structuretaxvaluedollarcnt'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# ## WE can remove the 46.40 ,which seeming outlier 

# In[58]:

imp_features.sort_values(by = 'structuretaxvaluedollarcnt', ascending = False)[:3]


# ### But after Seeing the data we notice that the building may be categorised for the antique as it is built in 1937 ,so we would avoid to delete the data

# In[59]:

### Normalization of the data 


# <p> We have chosen  Johnson distribution for better accuracy with same alignmnet as of the original data</p>

# In[60]:

imp_features['structuretaxvaluedollarcnt']


# In[61]:

import scipy.stats as st
y =imp_features['structuretaxvaluedollarcnt']
z=imp_features['calculatedfinishedsquarefeet']
x=imp_features['lotsizesquarefeet']
v=imp_features['structuretaxvaluedollarcnt']
u=imp_features['taxamount']


plt.title('Johnson SU')
plt.figure(1);sns.distplot(x, kde=False, fit=st.johnsonsu)
plt.figure(2);sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(3);sns.distplot(z, kde=False, fit=st.johnsonsu)
plt.figure(4);sns.distplot(u, kde=False, fit=st.johnsonsu)
plt.figure(5);sns.distplot(v, kde=False, fit=st.johnsonsu)


# In[62]:

fitw=st.lognorm(imp_features['structuretaxvaluedollarcnt'])


# In[63]:

print(fitw)


# In[64]:

y.head()


# In[65]:

imp_features['structuretaxvaluedollarcnt'].head()


# ### We would apply samefor other quantitatice columns available 

# In[73]:

quantitative = ['calculatedfinishedsquarefeet','lotsizesquarefeet','structuretaxvaluedollarcnt','taxamount']


f = pd.melt(imp_features, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


# In[74]:

imp_features.to_csv('cleanData.csv',mode='wb')


# In[76]:

imp_features.info()


# In[77]:

import datetime, time
#fetching the timestamp
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d%m%y%M%S')
st1 = datetime.datetime.fromtimestamp(ts).strftime('%d%m%y')


# In[78]:

# Create logfile.
logfile = open(st+".txt", "a")
def log_entry(s):
    #print('Date now: %s' % datetime.datetime.now())

    timestamp = '[%s] : ' % datetime.datetime.now()
    log_line = timestamp + s + '\n'
    logfile.write(log_line)
    logfile.flush()
log_entry("Import Done")


# In[80]:

with open('config.json') as data_file:    
    configdata = json.load(data_file)


# In[81]:

AWS_ACCESS_KEY_ID = configdata["AWSAccess"]
print(AWS_ACCESS_KEY_ID)
AWS_SECRET_ACCESS_KEY = configdata["AWSSecret"]
print(AWS_SECRET_ACCESS_KEY)
TeamNumber="9"
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


# In[83]:

# uploading Clean data to s3 bucket 
bucket_name = "team_9" + 'assignment2_clean'
print bucket_name
log_entry("S3 bucket has been successfully created.")
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
print conn
bucket = conn.create_bucket(bucket_name, location=boto.s3.connection.Location.DEFAULT)
filename_clean_data="team_9"+"_"+"clean"+"_"+st1
print filename_clean_data
imp_features.to_csv(filename_clean_data+".csv")

s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
file = filename_clean_data+".csv"
exists = False
s3Session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name='us-east-1'
)

try:
    s3.Object(bucket_name, file).load()
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        exists = False
    else:
        raise
else:
    exists = True

if exists==False:
    print ('Uploading %s to Amazon S3 bucket %s' % (file, bucket_name))
    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()
    k = Key(bucket)
    k.key = file
    k.set_contents_from_filename(file, cb=percent_cb, num_cb=10)
    log_entry(file+" has been uploaded to "+bucket_name)
    bucket_policy = s3.BucketPolicy(bucket_name)
    print("File uploaded.")
    dataDownloadLink = s3Session.client('s3').generate_presigned_url(
    ClientMethod='get_object',
    Params={
        'Bucket': bucket_name,
        'Key': file})

    
elif exists==True:
    print("File already exists.")
    log_entry("File already exists.")
    s3.Object(bucket_name, file).put(Body=open(file, 'rb'))
    s3.Object(bucket_name, file).Acl().put(ACL='public-read')
    dataDownloadLink = s3Session.client('s3').generate_presigned_url(
    ClientMethod='get_object',
    Params={
        'Bucket': bucket_name,
        'Key': file})


# In[84]:

try:
    s3Session.client('ses').send_email(
        Destination={
            'ToAddresses': [configdata["notificationEmail"]]
        },
        Message={
          'Body': {
              'Text': {
                  'Data': dataDownloadLink
              }
          },
          'Subject': {
              'Data': "File to Download"
          }
        },
        Source=configdata["notificationEmail"]
    )
except Exception as e:
    print(e)


# In[ ]:



