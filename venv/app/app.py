
# coding: utf-8

# ## Import Statements

# In[25]:

from flask import Flask,render_template,request
import seaborn as sns
import psycopg2
import boto3
import boto.ec2
import re
import pandas as pd
import time
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from jinja2.utils import Markup
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## Connecting to Ec2 Instance on AWS

# In[26]:

conn = boto.ec2.connect_to_region("us-east-1",aws_access_key_id="AKIAIQXRFJBYHUO2VEHA",aws_secret_access_key="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")


# In[27]:

hostaws = 'ec2-52-203-231-78.compute-1.amazonaws.com'


# ## Connecting to Postgres Sql deployed on EC2 Instance

# In[28]:

pgConn = psycopg2.connect(host=hostaws,user='postgres',database='postgres')
print ("Opening connection using DSN:"+str(hostaws))
print "Encoding for this connection is", pgConn.encoding


# In[29]:

app=Flask(__name__)


# In[30]:

@app.route('/send', methods=['POST','GET'])
def send():
    if request.method == 'POST':
        print ("POST")
        index=request.form['index']
        parcelid=request.form['parcelid']
        bathroomcnt=request.form['bathroomcnt']
        bedroomcnt=request.form['bedroomcnt']
        buildingqualitytypeid=request.form['buildingqualitytypeid']
        calculatedfinishedsquarefeet=request.form['calculatedfinishedsquarefeet']
        fips=request.form['fips']
        heatingorsystemtypeid=request.form['heatingorsystemtypeid']
        longitude=request.form['longitude']
        latitude=request.form['latitude']
        lotsizesquarefeet=request.form['lotsizesquarefeet']
        rawcensustractandblock=request.form['rawcensustractandblock']
        regionidcity=request.form['regionidcity']
        regionidcounty=request.form['regionidcounty']
        unitcnt=request.form['unitcnt']
        yearbuilt=request.form['yearbuilt']
        landtaxvaluedollarcnt=request.form['landtaxvaluedollarcnt']
        taxamount=request.form['taxamount']
        censustractandblock=request.form['censustractandblock']
        transactiondate=time.strftime("%x")
        print ("received")
        curs = pgConn.cursor()
        
        query =  "INSERT INTO properties (index, parcelid,bathroomcnt,bedroomcnt,buildingqualitytypeid,calculatedfinishedsquarefeet,fips,heatingorsystemtypeid,latitude,longitude,lotsizesquarefeet,rawcensustractandblock,regionidcity,regionidcounty,roomcnt,yearbuilt,structuretaxvaluedollarcnt,taxamount,censustractandblock,transactiondate) VALUES (%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s, %s, %s);"
        data = (index,parcelid,bathroomcnt,bedroomcnt,buildingqualitytypeid,calculatedfinishedsquarefeet,fips,heatingorsystemtypeid,latitude,longitude,lotsizesquarefeet,rawcensustractandblock,regionidcity,regionidcounty,unitcnt,yearbuilt,landtaxvaluedollarcnt,taxamount,censustractandblock,transactiondate)
        print("trying to print data")
        curs.execute(query, data)
        print("data inserted")
        pgConn.commit()
        curs.close()       
        return render_template('create.html', Status="data inserted")
    return render_template('create.html')


# In[31]:

@app.route('/fetch', methods=['POST','GET'])
def fetch():
    if request.method == 'POST':
        print ("POST")
        index=request.form.get('selectList')
        print(index)
        value=request.form['value']
        print(value)
        operand=request.form.get('operand')
        print(operand)
        curs = pgConn.cursor()
        query = "SELECT * FROM properties where "+ index+operand+value+";"
        curs.execute(query)
        records = curs.fetchall()
        dataf = pd.DataFrame(records)
        dataf.columns=['index','parcelid','bathroomcnt','bedroomcnt','buildingqualitytypeid','calculatedfinishedsquarefeet','fips','heatingorsystemtypeid','latitude','longitude','lotsizesquarefeet','rawcensustractandblock','regionidcity','regionidcounty','unitcnt','yearbuilt','landtaxvaluedollarcnt','taxamount','censustractandblock','transactiondate']
        print(dataf.head())
        pgConn.commit()
        curs.close()  
        titles = ['na']      
        return render_template('fetch.html',tables=[dataf.to_html(classes='table table-striped')], titles=titles)
       
    return render_template('fetch.html')

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('fetch.html',Status="No records found"), 500


# In[32]:

@app.route('/results', methods=['GET', 'POST'])
def results():
    error = None
    if request.method == 'POST':
        curs = pgConn.cursor()
        curs.execute('SELECT * FROM properties;')
        records = curs.fetchall()
        dataf = pd.DataFrame(records)
        dataf.columns=['index', 'parcelid','bathroomcnt','bedroomcnt','buildingqualitytypeid','calculatedfinishedsquarefeet',
                       'fips','heatingorsystemtypeid','latitude','longitude','lotsizesquarefeet','rawcensustractandblock'
                       ,'regionidcity','regionidcounty','roomcnt','yearbuilt','structuretaxvaluedollarcnt','taxamount',
                       'censustractandblock','transactiondate']
        
        
        dataf['transactiondate']=pd.to_datetime(dataf['transactiondate'],format= '%Y-%m-%d')
        q=dataf['transactiondate'].dt.month.value_counts()
        q=dict(q)
        m=[]
        f=[]
        for the_key, the_value in q.iteritems():
            m.append(the_key)
            f.append(q[the_key])
        #plt.bar(m,f,  align='center')
        #plt.title("frequency of transactions each month")
        #my_plot_div = plt.show()      
        
        data = [go.Bar(x=m, y=f)]

        url = plot(data, filename='frequency of transactions each month', output_type='div')
             
        sns.jointplot(y=dataf['calculatedfinishedsquarefeet'] ,x=dataf['bathroomcnt'])         
        
       # my_plot_div = plot([Scatter(x=m, y=f)], output_type='div')
        return render_template('eda.html',
                               Graph1=Markup(url),Graph2=Markup(url2)
                              )
    elif request.method == "GET":
        return render_template('eda.html')


# In[33]:

@app.route('/delete', methods=['POST','GET'])
def delete():
    if request.method == 'POST':
        print ("POST")
        index=request.form.get('selectList')
        print(index)
        value=request.form['value']
        print(value)
        operand=request.form.get('operand')
        print(operand)
        curs = pgConn.cursor()
        query = "DELETE FROM properties where "+ index+operand+value+";"
        curs.execute(query)
        pgConn.commit()
        curs.close()  
        return render_template('delete.html',Status="Record Deleted")
       
    return render_template('delete.html')


# In[34]:

from geopy.distance import great_circle

@app.route('/top10', methods=['POST','GET'])
def top10():
    if request.method == 'POST':
        curs = pgConn.cursor()
        curs.execute('SELECT * FROM properties;')
        records = curs.fetchall()
        dataf = pd.DataFrame(records)
        dataf.columns=['index', 'parcelid','bathroomcnt','bedroomcnt','buildingqualitytypeid','calculatedfinishedsquarefeet',
                       'fips','heatingorsystemtypeid','latitude','longitude','lotsizesquarefeet','rawcensustractandblock'
                       ,'regionidcity','regionidcounty','roomcnt','yearbuilt','structuretaxvaluedollarcnt','taxamount',
                       'censustractandblock','transactiondate']
        latitude=request.form['latitude']
        longitude=request.form['longitude']
        input1=latitude
        input2=longitude
        userInput=(input1,input2)
        var={'parcelid':[],'longitude':[],'latitude':[],'distance':[]}
        for i,r in dataf.iterrows():
            lo =  r['longitude']
            la =  r['latitude']
            if(input1==lo):
                if(input2==lo):
                    var['parcelid'].append(r['parcelid'])
                    var['longitude'].append(r['longitude'])
                    var['latitude'].append(r['latitude'])
            else:
                    finder=(lo,la)
                    distance = great_circle(userInput, finder).miles
                    var['parcelid'].append(r['parcelid'])
                    var['longitude'].append(r['longitude'])
                    var['latitude'].append(r['latitude'])
                    var['distance'].append(distance)

        p = pd.DataFrame(var) 
        ans = p.sort_values('distance',ascending=True).head(10)
        ans = ans.reset_index()
        titles = ['na']
        
        #return render_template('fetch.html',tables=[dataf.to_html(classes='table table-striped')], titles=titles)
        return render_template('top10.html',tables=[ans.to_html(classes='table table-striped')], titles=titles)
    return render_template('top10.html')


# In[ ]:

if __name__=="__main__":
    app.run()


# In[ ]:




# In[ ]:



