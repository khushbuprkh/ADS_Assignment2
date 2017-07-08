
# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np
from datetime import datetime
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
 



# In[6]:

properties =pd.read_csv('properties_2016.csv',low_memory=False)
train =pd.read_csv('train_2016_v2.csv')
print('Step 1. Import of Data Done')


# In[7]:

properties.info()


# In[14]:

properties["structuretaxvaluedollarcnt"].unique()


# In[10]:

merged_data=pd.merge(properties,train,on='parcelid',how='left')


# In[ ]:

### AS the data after the October , we dont have any value for the columns except the parcelid column, this shows we dont have data fater the october month.


# In[ ]:

### So we would remove the data after the october


# In[11]:

merged_data=merged_data[(merged_data['transactiondate'] <='2016-10-01')]


# In[12]:

merged_data.info()


# In[ ]:

## WE dont find any junk characters present in the Data So we would Drop the values which are Null 


# In[ ]:




# In[ ]:

import datetime, time
#fetching the timestamp
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d%m%y%M%S')
st1 = datetime.datetime.fromtimestamp(ts).strftime('%d%m%y')


# In[4]:

# Create logfile.
logfile = open(st+".txt", "a")
def log_entry(s):
    #print('Date now: %s' % datetime.datetime.now())

    timestamp = '[%s] : ' % datetime.datetime.now()
    log_line = timestamp + s + '\n'
    logfile.write(log_line)
    logfile.flush()
log_entry("Import Done")


# In[ ]:

with open('config.json') as data_file:    
    configdata = json.load(data_file)


# In[ ]:

AWS_ACCESS_KEY_ID = configdata["AWSAccess"]
print(AWS_ACCESS_KEY_ID)
AWS_SECRET_ACCESS_KEY = configdata["AWSSecret"]
print(AWS_SECRET_ACCESS_KEY)
TeamNumber="9"
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


# In[ ]:

# uploading Clean data to s3 bucket 
bucket_name = "team_9" + 'assignment2_clean'
print bucket_name
log_entry("S3 bucket has been successfully created.")
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
print conn
bucket = conn.create_bucket(bucket_name, location=boto.s3.connection.Location.DEFAULT)
filename_clean_data="team_9"+"_"+"clean"+"_"+st1
print filename_clean_data
merged_data.to_csv(filename_clean_data+".csv")

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



# In[ ]:

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




# In[ ]:



