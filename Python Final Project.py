#!/usr/bin/env python
# coding: utf-8

# # NJ PPP Loan Data Analysis
# <b>Data acquired from <a href="https://data.nj.gov/Government-Finance/PPP-Cares-Act-Loan-Totals-to-New-Jersey-Businesses/riep-z5cp">nj.data.gov</a>.</b>

# ## Background
# <p>
#     The Paycheck Protection Program (PPP) loans provide small businesses with the resources they need to maintain their payroll, hire back employees who may have been laid off, and cover applicable overhead. This data set includes businesses in New Jersey who received PPP funding, how much funding the employer received & how many jobs the employer claims they saved. The NAICS (National Industry Classification) was provided by the loan recipient.
# </p>
# <p>
#     This dataset was used to analyze the distribution of Payment Protection Program loans within New Jersey. In this notebook is a breakdown consisting of loan bracket, business owner race, and other important distinctions within PPP loan distribution. 
# </p>

# In[1]:


import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns
import numpy as np

import matplotlib as mpl
from matplotlib import rcParams
from matplotlib import style
mpl.style.use('fivethirtyeight')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import csv
import math
import json


# ### General Overview

# In[2]:


INIT_CSV = pd.DataFrame(pd.read_csv('PPP_Data_fixed - Final.csv'))
print('\n')
profile = ProfileReport(INIT_CSV, title='CSV Pandas Profiling Report', explorative = True, minimal = True)


# In[3]:


profile.to_widgets()


# ## Data Preview

# In[4]:


data = INIT_CSV.copy()
data.transpose()


# ### Businesses owned by a White/Caucasian Individual

# In[5]:


WhtOwned = INIT_CSV.copy()
whtNum = WhtOwned[WhtOwned['Race Ethnicity'].str.contains("White")]
num = str(whtNum.shape[0])
print("There are " + num + " White owned businesses")
whtNum.head()


# ### Businesses owned by a Black/African American Individual

# In[6]:


AfrOwned = INIT_CSV.copy()
blkNum = AfrOwned[AfrOwned['Race Ethnicity'].str.contains("Black")]
num = str(blkNum.shape[0])
print("There are " + num + " Black/African American owned businesses")
blkNum.head()


# ### Businesses owned by an Asian Individual

# In[7]:


AsianOwned = INIT_CSV.copy()
asianNum = AsianOwned[AsianOwned['Race Ethnicity'].str.contains("Asian")]
num = str(asianNum.shape[0])
print("There are " + num + " Asian owned businesses")
asianNum.head()


# ### How many jobs were retained per industry?

# In[8]:


df = INIT_CSV.copy()
df = df[["NAICS Code", "Jobs Retained"]]

# Remove rows with empty data
df.dropna(inplace = True)

# Make sorted list of unique NAICS codes
codes = sorted(list(set(np.array([238220, 541110, 541511, 621111, 722511]))))
decodes = sorted(list(set(np.array(["Plumbing/HVAC\n238220", "Law Offices\n541110", "Computer Programming\n541511", "Physicians\n621111", "Restaurants\n722511"]))))
jobs = []

# Sum jobs retained for each job code
for code in codes:
    total = df.loc[df["NAICS Code"] == code, "Jobs Retained"].sum()
    jobs.append(total)
    
# Make ticks evenly spaced despite their values
x_pos = np.arange(len(codes))

fig, ax = plt.subplots(figsize=(8,3))

# Add chart labels
plt.barh(x_pos, jobs, .35)
plt.title("Jobs Retained Per Industry")
plt.ylabel("NAICS Code")
plt.xlabel("Jobs Retained")

plt.yticks(x_pos, decodes)
plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha="right", rotation_mode="anchor") 

for p in ax.patches:
#     ax.annotate(str(p.get_x()), (p.get_width() * 1.005, p.get_x() * 1.005))
    ax.annotate(str(p.get_width().astype(int)), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 1), textcoords='offset points')
    
plt.show()


# ### How many businesses took loans in each bracket?

# In[9]:


pd.set_option('precision', 0)
fields = ['Loan Range', 'NAICS Code']
order = ['$5,000,000-$10,000,000', '$2,000,000-$5,000,000', '$1,000,000-$2,000,000', '$350,000-$1,000,000', '$150,000-$350,000']
dfJob = pd.DataFrame(pd.read_csv('PPP_Data_fixed - Final.csv', skipinitialspace=True, usecols=fields))
ax = dfJob.groupby('Loan Range').size().reindex(order).plot(kind='barh', alpha=0.75, figsize=(8,3),      title="# of Businesses in Each Loan Bracket",      xlabel="Loan Amount", width=.35)
plt.gca().invert_yaxis()
for p in ax.patches:
    ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, -8), textcoords='offset points')
plt.xlabel("# of Businesses")
plt.tight_layout()


# ### Exploring specific cities in New Jersey

# In[10]:


cities = {}

with open('PPP_Data_fixed - Final.csv', 'r') as f_in:
    reader = csv.reader(f_in)
    count = 0
    for row in reader:
        if len(row) == 17:
            string = row[3]
        elif len(row) == 19:
            string = row[5]
        elif len(row) == 18:
            string = row[4]
        
        if string not in cities:
            cities[string] = 1
        else:
            cities[string] += 1
        count += 1
f_in.close()

city = input("Which city would you like to explore? ")
city = city.upper()
print('\n' + city.title() + " had " + str(cities.get(city.upper())) + " businesses recieve a PPP loan.")

#    df[(df['A'] > 1) | (df['B'] < -1)]

AfrOwned = INIT_CSV.copy()
blkNum = AfrOwned[(AfrOwned['Race Ethnicity'] == 'Black or African American') & (AfrOwned['City'] == city)]
num = str(blkNum.shape[0])
print(num + " of which are Black/African American owned businesses")

WhtOwned = INIT_CSV.copy()
whtNum = WhtOwned[(WhtOwned['Race Ethnicity'] == 'White') & (WhtOwned['City'] == city)]
num = str(whtNum.shape[0])
print(num + " of which are White owned businesses")

AsianOwned = INIT_CSV.copy()
asianNum = AsianOwned[(AsianOwned['Race Ethnicity'] == 'Asian') & (AsianOwned['City'] == city)]
num = str(asianNum.shape[0])
print(num + " of which are Asian owned businesses")

HispOwned = INIT_CSV.copy()
hispNum = HispOwned[(HispOwned['Race Ethnicity'] == 'Hispanic') & (HispOwned['City'] == city)]
num = str(hispNum.shape[0])
print(num + " of which are Hispanic owned businesses")

NativeOwned = INIT_CSV.copy()
natNum = NativeOwned[(NativeOwned['Race Ethnicity'] == 'American Indian or Alaska Native') & (NativeOwned['City'] == city)]
num = str(natNum.shape[0])
print(num + " of which are American Indian/Alaskan Native owned businesses")

NaNOwned = INIT_CSV.copy()
NaNNum = NaNOwned[(NaNOwned['Race Ethnicity'] == 'Unanswered') & (NaNOwned['City'] == city)]
num = str(NaNNum.shape[0])
print(num + " businesses did not answer")


# In[ ]:




