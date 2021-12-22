import pandas as pd
from datetime import datetime
import numpy as np
'''
Read Data
'''
df1 = pd.read_csv('marketing_campaign.csv')

'''
Remove Outliers
df1['Income']>150000 & df1['MntMeatProducts']>1000 & df1['MntSweetProducts']>200 & df1['MntGoldProds']>260
'''
df1 = df1.drop(df1[df1['Income']>150000].index)
df1 = df1.drop(df1[df1['MntMeatProducts']>1000].index)
df1 = df1.drop(df1[df1['MntSweetProducts']>200].index)
df1= df1.drop(df1[df1['MntGoldProds']>260].index)
df1 = df1.dropna()
df1 = df1.reset_index(drop=True)
available_rows = df1.shape[0]
'''
New column: age
'''
age = 2017 - df1.Year_Birth
age = pd.DataFrame(age.to_numpy(), columns=['Age'])

'''
New column: Number of day customer enroll with the company
'''
Dt_Customer = df1.Dt_Customer.to_numpy()
since_day = datetime(2017, 1, 1)
mapper1 = map(lambda x: since_day - datetime.strptime(x,"%d-%m-%Y"), Dt_Customer)
Days_since_enroll_with = pd.DataFrame(list(map(lambda x: x.days, mapper1)), columns=['NumOfDayEnrolled'])

'''
Modified column: martial status
'''
one_hot_martial_staus = pd.DataFrame(data=np.zeros((available_rows,6),dtype=int),columns=['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Other'])
for index ,status in enumerate(df1.Marital_Status):
    if status not in ['Married', 'Together', 'Single', 'Divorced', 'Widow']:
        status = 'Other'
    one_hot_martial_staus.loc[index, status] = 1


'''
Modified column: Education
'''
one_hot_Education = pd.DataFrame(data=np.zeros((available_rows,4),dtype=int),columns=['Basic','Graduation', 'Master', 'PhD'])
for index ,status in enumerate(df1.Education):
    if status == '2n Cycle':
        status = 'Master'
    one_hot_Education.loc[index, status] = 1

'''
Assemble basic information
'''
data = df1[['Income','Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
data = pd.concat([data, age, Days_since_enroll_with, one_hot_martial_staus, one_hot_Education],axis=1)

data.to_csv('basic_info.csv',index=False)


