#!/usr/bin/env python
# coding: utf-8

# %autosave 0

# In[ ]:





# # Introduction to Pandas
# 
# In this section of the course we will learn how to use pandas for data analysis. You can think of pandas as an extremely powerful version of Excel, with a lot more features. In this section of the course, you should go through the notebooks in this order:
# 
# * Introduction to Pandas
# * Series
# * DataFrames
# * Missing Data
# * GroupBy
# * Merging,Joining,and Concatenating
# * Operations
# * Data Input and Output

# # Series
# The first main data type we will learn about for pandas is the Series data type. Let's import Pandas and explore the Series object
# 
# A Series is very similar to a NumPy array (in fact it is built on top of the NumPy array object). What differentiates the NumPy array from a Series, is that a Series can have axis labels, meaning it can be indexed by a label, instead of just a number location. It also doesn't need to hold numeric data, it can hold any arbitrary Python Object.
# 
# Let's explore this concept through some examples:

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


### Creating a Series
#You can convert a list,numpy array, or dictionary to a Series:


# In[4]:


labels = ['a','b','c']
my_list = [10,20,30]
arr = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}


# In[6]:


#Using Lists
pd.Series(data=my_list)


# In[7]:


pd.Series(data=my_list,index=labels)


# In[8]:


pd.Series(my_list,labels)


# In[9]:


#** NumPy Arrays **


# In[10]:


pd.Series(arr)


# In[11]:


pd.Series(arr,labels)


# In[12]:


#** Dictionary**


# In[13]:


pd.Series(d)


# In[14]:


### Data in a Series
#A pandas Series can hold a variety of object types:


# In[15]:


pd.Series(data=labels)


# In[16]:


# Even functions (although unlikely that you will use this)
pd.Series([sum,print,len])


# In[18]:


## Using an Index

#The key to using a Series is understanding its index. Pandas makes use of these index names or numbers by allowing for fast look ups of information (works like a hash table or dictionary).

#Let's see some examples of how to grab information from a Series. Let us create two sereis, ser1 and ser2:


# In[19]:


ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])                                   


# In[20]:


ser1


# In[21]:


ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])                                   


# In[22]:


ser2


# In[23]:


ser1['USA']


# In[24]:


#Operations are then also done based off of index:


# In[25]:


ser1 + ser2


# In[27]:


# DataFrames

#DataFrames are the workhorse of pandas and are directly inspired by the R programming language. We can think of a DataFrame as a bunch of Series objects put together to share the same index. Let's use pandas to explore this topic!


# In[28]:


from numpy.random import randn
np.random.seed(101)


# In[29]:


df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())


# In[30]:


df


# In[31]:


## Selection and Indexing

#Let's learn the various methods to grab data from a DataFrame


# In[32]:


df['W']


# In[33]:


# Pass a list of column names
df[['W','Z']]


# In[34]:


# SQL Syntax (NOT RECOMMENDED!)
df.W


# In[35]:


#DataFrame Columns are just Series


# In[36]:


type(df['W'])


# In[37]:


#**Creating a new column:**


# In[38]:


df['new'] = df['W'] + df['Y']


# In[39]:


df


# In[40]:


#** Removing Columns**


# In[41]:


df.drop('new',axis=1)


# In[42]:


# Not inplace unless specified!
df


# In[43]:


df.drop('new',axis=1,inplace=True)


# In[44]:


df


# In[45]:


#Can also drop rows this way:


# In[46]:


df.drop('E',axis=0)


# In[47]:


#** Selecting Rows**


# In[48]:


df.loc['A']


# In[49]:


#Or select based off of position instead of label 


# In[50]:


df.iloc[2]


# In[51]:


#** Selecting subset of rows and columns **


# In[52]:


df.loc['B','Y']


# In[53]:


df.loc[['A','B'],['W','Y']]


# In[54]:


### Conditional Selection
#An important feature of pandas is conditional selection using bracket notation, very similar to numpy:


# In[55]:


df


# In[56]:


df>0


# In[57]:


df[df>0]


# In[58]:


df[df['W']>0]


# In[59]:


df[df['W']>0]['Y']


# In[60]:


df[df['W']>0][['Y','X']]


# In[61]:


#For two conditions you can use | and & with parenthesis:


# In[62]:


df[(df['W']>0) & (df['Y'] > 1)]


# In[63]:


## More Index Details

#Let's discuss some more features of indexing, including resetting the index or setting it something else. We'll also talk about index hierarchy!


# In[64]:


# Reset to default 0,1...n index
df.reset_index()


# In[65]:


newind = 'CA NY WY OR CO'.split()


# In[66]:


df['States'] = newind


# In[67]:


df


# In[68]:


df.set_index('States')


# In[69]:


df


# In[70]:


df.set_index('States',inplace=True)


# In[71]:


## Multi-Index and Index Hierarchy
#Let us go over how to work with Multi-Index, first we'll create a quick example of what a Multi-Indexed DataFrame would look like:


# In[73]:


# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)


# In[74]:


hier_index


# In[75]:


df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df


# In[76]:


#Now let's show how to index this! For index hierarchy we use df.loc[], if this was on the columns axis, you would just use normal bracket notation df[]. Calling one level of the index returns the sub-dataframe:


# In[77]:


df.loc['G1']


# In[78]:


df.loc['G1'].loc[1]


# In[79]:


df.index.names


# In[80]:


df.index.names = ['Group','Num']


# In[81]:


df


# # Missing Data
# #Let's show a few convenient methods to deal with Missing Data in pandas:

# In[82]:


import numpy as np
import pandas as pd


# In[84]:


df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})


# In[85]:


df


# In[86]:


df.dropna()


# In[87]:


df.dropna(axis=1)


# In[88]:


df.dropna(thresh=2)


# In[89]:


df.fillna(value='FILL VALUE')


# In[90]:


df['A'].fillna(value=df['A'].mean())


# # Groupby
# 
# The groupby method allows you to group rows of data together and call aggregate functions

# In[91]:


import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}


# In[92]:


df = pd.DataFrame(data)


# In[93]:


df


# ** Now you can use the .groupby() method to group rows together based off of a column name. For instance let's group based off of Company. This will create a DataFrameGroupBy object:**

# In[94]:


df.groupby('Company')


# You can save this object as a new variable:

# In[95]:


by_comp = df.groupby("Company")


# And then call aggregate methods off the object:

# In[96]:


by_comp.mean()


# In[97]:


df.groupby('Company').mean()


# More examples of aggregate methods:|

# In[98]:


by_comp.std()


# In[99]:


by_comp.min()


# In[100]:


by_comp.max()


# In[101]:


by_comp.count()


# In[102]:


by_comp.describe()


# In[103]:


by_comp.describe().transpose()


# In[104]:


by_comp.describe().transpose()['GOOG']


# # Merging, Joining, and Concatenating
# 
# There are 3 main ways of combining DataFrames together: Merging, Joining and Concatenating. In this lecture we will discuss these 3 methods with examples.

# In[106]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])


# In[107]:


df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 


# In[108]:


df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])


# In[109]:


df1


# In[110]:


df2


# In[111]:


df3


# ## Concatenation
# 
# Concatenation basically glues together DataFrames. Keep in mind that dimensions should match along the axis you are concatenating on. You can use **pd.concat** and pass in a list of DataFrames to concatenate together:

# In[112]:


pd.concat([df1,df2,df3])


# In[113]:


pd.concat([df1,df2,df3],axis=1)


# In[114]:


left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})    


# In[115]:


left


# In[116]:


right


# ## Merging
# 
# The **merge** function allows you to merge DataFrames together using a similar logic as merging SQL Tables together. For example:

# In[117]:


pd.merge(left,right,how='inner',on='key')


# In[118]:


left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})


# In[119]:


pd.merge(left, right, on=['key1', 'key2'])


# In[120]:


pd.merge(left, right, how='outer', on=['key1', 'key2'])


# In[121]:


pd.merge(left, right, how='right', on=['key1', 'key2'])


# In[122]:


pd.merge(left, right, how='left', on=['key1', 'key2'])


# ## Joining
# Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame.

# In[124]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])


# In[125]:


left.join(right)


# In[126]:


left.join(right, how='outer')


# # Operations
# 
# There are lots of operations with pandas that will be really useful to you, but don't fall into any distinct category. Let's show them here in this lecture:

# In[127]:


df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()


# In[128]:


### Info on Unique Values


# In[129]:


df['col2'].unique()


# In[130]:


df['col2'].nunique()


# In[131]:


df['col2'].value_counts()


# In[132]:


# Selecting Data


# In[134]:


#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]


# In[135]:


newdf


# In[136]:


### Applying Functions


# In[137]:


def times2(x):
    return x*2


# In[138]:


df['col1'].apply(times2)


# In[139]:


df['col3'].apply(len)


# In[140]:


df['col1'].sum()


# ** Permanently Removing a Column**

# In[141]:


del df['col1']


# In[142]:


df


# ** Get column and index names: **

# In[143]:


df.columns


# In[144]:


df.index


# ** Sorting and Ordering a DataFrame:**

# In[145]:


df


# In[146]:


df.sort_values(by='col2') #inplace=False by default


# ** Find Null Values or Check for Null Values**

# In[147]:


df.isnull()


# In[148]:


# Drop rows with NaN Values
df.dropna()


# ** Filling in NaN values with something else: **

# In[149]:


df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()


# In[150]:


df.fillna('FILL')


# In[151]:


data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)


# In[152]:


df


# In[153]:


df.pivot_table(values='D',index=['A', 'B'],columns=['C'])


# # Data Input and Output
# 
# This notebook is the reference code for getting input and output, pandas can read a variety of file types using its pd.read_ methods. Let's take a look at the most common data types:

# In[154]:


df = pd.read_csv('example')
df


# ### CSV Output

# In[155]:


df.to_csv('example',index=False)


# ## Excel
# Pandas can read and write excel files, keep in mind, this only imports data. Not formulas or images, having images or macros may cause this read_excel method to crash. 

# ### Excel Input

# In[159]:


pd.read_excel('Excel_Sample.xlsx',index_col=0)


# ## HTML
# 
# You may need to install htmllib5,lxml, and BeautifulSoup4. In your terminal/command prompt run:
# 
#     conda install lxml
#     conda install html5lib
#     conda install BeautifulSoup4
# 
# Then restart Jupyter Notebook.
# (or use pip install if you aren't using the Anaconda Distribution)
# 
# Pandas can read table tabs off of html. For example:

# ### HTML Input
# 
# Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects:

# In[162]:


df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')


# In[164]:


df[0]


# # SQL (Optional)
# 
# * Note: If you are completely unfamiliar with SQL you can check out my other course: "Complete SQL Bootcamp" to learn SQL.

# The pandas.io.sql module provides a collection of query wrappers to both facilitate data retrieval and to reduce dependency on DB-specific API. Database abstraction is provided by SQLAlchemy if installed. In addition you will need a driver library for your database. Examples of such drivers are psycopg2 for PostgreSQL or pymysql for MySQL. For SQLite this is included in Python’s standard library by default. You can find an overview of supported drivers for each SQL dialect in the SQLAlchemy docs.
# 
# 
# If SQLAlchemy is not installed, a fallback is only provided for sqlite (and for mysql for backwards compatibility, but this is deprecated and will be removed in a future version). This mode requires a Python database adapter which respect the Python DB-API.
# 
# See also some cookbook examples for some advanced strategies.
# 
# The key functions are:
# 
# * read_sql_table(table_name, con[, schema, ...])	
#     * Read SQL database table into a DataFrame.
# * read_sql_query(sql, con[, index_col, ...])	
#     * Read SQL query into a DataFrame.
# * read_sql(sql, con[, index_col, ...])	
#     * Read SQL query or database table into a DataFrame.
# * DataFrame.to_sql(name, con[, flavor, ...])	
#     * Write records stored in a DataFrame to a SQL database.

# In[165]:


from sqlalchemy import create_engine


# In[166]:


engine = create_engine('sqlite:///:memory:')


# In[ ]:





# In[ ]:




