#!/usr/bin/env python
# coding: utf-8

# ## 1. Import required libraries and read the dataset.

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


play = pd.read_csv(r"E:\GLCA_AUG_2023\Python_Files\python project\project_4\Apps_data+(1).csv")


# ## 2. Check the first few samples, shape, info of the data and try to familiarize yourself with different features.

# In[5]:


play.head(5)


# In[6]:


play.shape


# In[7]:


play.info()


# ## 3. Check summary statistics of the dataset. List out the columns that need to be worked upon for model building.

# In[8]:


play.describe(include="all")


# In[9]:


play.nunique()


# ## 4. Check if there are any duplicate records in the dataset? if any drop them.

# In[10]:


play[play.duplicated()]


# In[11]:


play.drop_duplicates(inplace=True)


# In[12]:


play[play.duplicated()]


# ## 5. Check the unique categories of the column 'Category', Is there any invalid category? If yes, drop them.
# 

# In[13]:


play.drop(play.query("Category == '1.9'").index , inplace= True)


# In[14]:


play.Category.unique()


# In[15]:


play[play.Category == '1.9']


# In[ ]:





# In[ ]:





# ## 6. Check if there are missing values present in the column Rating, If any? drop them and and create a new column as 'Rating_category' by converting ratings to high and low categories(>3.5 is high rest low).

# In[ ]:





# In[16]:


play.isna().sum()


# In[17]:


play.replace(" ", np.nan, inplace = True)


# In[18]:


play.dropna(inplace=True)


# In[19]:


play[play.Rating == ' ']


# In[20]:


play["Rating_Category"] = play.Rating.apply(lambda x: "High" if x > 3.5 else "LOW")


# In[21]:


play.head(10)


# In[ ]:





# In[ ]:





# ## 7. Check the distribution of the newly created column 'Rating_category' and comment on the distribution.

# In[22]:


play.Rating_Category.unique()


# In[23]:


play.Rating_Category.value_counts()


# In[24]:


#There are 8007 ratings which are greater than 3.5 ratings given by custemors 
#and 879 low rating which are below 3.5 given by customers 


# In[ ]:





# ## 8. Convert the column "Reviews'' to numeric data type and check the presence of outliers in the column and handle the outliers using a transformation approach.(Hint: Use log transformation)
# 

# In[25]:


play["Reviews"] = play["Reviews"].astype(int)


# In[26]:


play.Reviews.dtype


# In[27]:


play.Reviews.plot(kind="box")


# In[28]:


play["Reviews"] = np.log10(play["Reviews"])


# In[29]:


play.head(10)


# In[30]:


play.Reviews.plot(kind="box")


# In[ ]:





# ## 9. The column 'Size' contains alphanumeric values, treat the non numeric data and convert the column into suitable data type. (hint: Replace M with 1 million and K with 1 thousand, and drop the entries where size='Varies with device')
# 

# In[31]:


play.Size.dtype


# In[32]:


play["Size"] = play["Size"].str.replace("M",'000000')


# In[33]:


play["Size"] = play["Size"].str.replace("k",'000')


# In[34]:


play


# In[35]:


play.drop(play.query("Size == 'Varies with device'").index , inplace= True)


# In[36]:


play["Size"] = play["Size"].astype(float)


# In[37]:


play.Size.dtype


# In[38]:


play.head(10)


# In[ ]:





# In[ ]:





# ## 10. Check the column 'Installs', treat the unwanted characters and convert the column into a suitable data type.

# In[39]:


play["Installs"] = play.Installs.str.replace(",",'')
play["Installs"] = play.Installs.str.replace("+",'')


# In[40]:


play.head(10)


# In[41]:


play["Installs"] = play["Installs"].astype(int)


# In[42]:


play.Installs.dtype


# In[ ]:





# ## 11. Check the column 'Price' , remove the unwanted characters and convert the column into a suitable data type.

# In[43]:


play.Price.unique()


# In[44]:


play["Price"] = play["Price"].str.replace("$",'')


# In[45]:


play["Price"] = play["Price"].astype(float)


# In[ ]:





# ## 12. Drop the columns which you think redundant for the analysis.(suggestion: drop column 'rating', since we created a new feature from it (i.e. rating_category) and the columns 'App', 'Rating' ,'Genres','Last Updated', 'Current Ver','Android Ver' columns since which are redundant for our analysis)
# 

# In[46]:


play.head()


# In[47]:


play.drop(['App','Rating' ,'Genres','Last Updated', 'Current Ver','Android Ver'], axis=1,inplace=True)


# In[48]:


play.head()


# In[49]:


play.nunique()


# In[ ]:





# ## 13. Encode the categorical columns.
# 

# In[50]:


play.head()


# In[53]:


play1 = pd.get_dummies(play,columns=["Category", "Content Rating"])


# In[54]:


play1.head()


# In[57]:


from sklearn.preprocessing import LabelEncoder


# In[58]:


LEncoder = LabelEncoder()


# In[59]:


play1["Type"] = LEncoder.fit_transform(play1["Type"])


# In[61]:


play1["Rating_Category"] = LEncoder.fit_transform(play1["Rating_Category"])


# In[63]:


play1.head()


# In[ ]:





# In[ ]:





# ## 14. Segregate the target and independent features (Hint: Use Rating_category as the target)
# 

# In[65]:


from sklearn.model_selection import train_test_split as tts


# In[73]:


x = play1.drop(["Rating_Category"], axis=1)


# In[75]:


y = play1[["Rating_Category"]]


# In[ ]:





# ## 15. Split the dataset into train and test.
# 

# In[76]:


x_train, x_test, y_train, y_test = tts(x,y, random_state=1, test_size=0.20)


# In[ ]:





# ## 16. Standardize the data, so that the values are within a particular range.
# 

# In[81]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()


# In[ ]:





# In[ ]:





# In[82]:


data = scale.fit_transform(play1)


# In[84]:


standard = pd.DataFrame(data, columns=play1.columns)


# In[85]:


X =standard.drop(['Rating_Category'], axis=1)


# In[86]:


Y = play1[['Rating_Category']]


# In[87]:


X_train, X_test, Y_train, Y_test = tts(X,Y, random_state=1, test_size=0.20)


# In[ ]:





# In[ ]:





# In[88]:


from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()


# In[89]:


regression_model.fit(X_train, Y_train)


# In[ ]:




