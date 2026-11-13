from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#استفاده از متد fetch_usirepo برای لود کردن دیتاست انتخابی با توجه به آیدی گفته شده در سایت مرود نظر
iranian_churn = fetch_ucirepo(id=563)


#استفاده از متد read_csv از پکیج پانداس برای خواندن فایل دیتاست انتخابی
df = pn.read_csv("CustomerChurn.csv")

#مشخصه های آماری



print('\n'"----------------------")
print("Feature 1 = Call  Failure")

#انتخاب ستون یا ویژگی برای استخراج مشخصه های آماری از طریق متد tolist() پیکیج پانداس
# برای تبدیل داده ها به ساختمان داده لیست



column_list = df["Call  Failure"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 2 = Complains")


column_list = df["Complains"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 3 = Subscription  Length")


column_list = df["Subscription  Length"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 4 = Charge  Amount")


column_list = df["Charge  Amount"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')



print('\n'"-------------------")
print("Feature 5 = Seconds of Use")


column_list = df["Seconds of Use"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 6 = Frequency of use")


column_list = df["Frequency of use"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')



print('\n'"-------------------")
print("Feature 7 = Frequency of SMS")
 

column_list = df["Frequency of SMS"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 8 = Distinct Called Numbers")
 

column_list = df["Distinct Called Numbers"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 9 = Age Group")


column_list = df["Age Group"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 10 = Tariff Plan")


column_list = df["Tariff Plan"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 11 = Status")
 

column_list = df["Status"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 12 = Age")


column_list = df["Age"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




print('\n'"-------------------")
print("Feature 13 = Customer Value")


column_list = df["Customer Value"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')



print('\n'"-------------------")
print("Class label = Churn")


column_list = df["Churn"].tolist()

#نمایش تعداد نمونه برای ویژگی مشخص شده
print(f'length is : {len(column_list)}')

#نمایش میانگین داده های ویژگی مشخص شده
print(f'mean is : {np.mean(column_list)}')

#نمایش میانه یا مقدار وسطی داده های ویژگی مشخص شده
print(f'median is : {np.median(column_list)}')

#نمایش مد یا بیشترین فراوانی داده های ویژگی مشخص شده
print(f'mode is : {stats.mode(column_list , keepdims=False)}')

#نمایش واریانس داده های ویژگی مشخص شده
print(f'Variance is : {np.var(column_list)}')

#نمایش انحراف معیار داده های ویژگی مشخص شده
print(f'standard deviation is : {np.std(column_list)}')




#Boxplot Complains by Call  Failure


#Boxplot Complains by Call Failure
plt.figure(figsize=(8, 4) , dpi=100)
sns.boxplot(x='Complains' , y='Call  Failure' , data=df)
plt.title(' Complains by Call Failure')
plt.xlabel('Complains')
plt.ylabel('Call  Failure')
plt.xticks(rotation=45)
plt.show()



#Boxplot Subscription  Length by Age
plt.figure(figsize=(8, 4) , dpi=100)
sns.boxplot(x='Age' , y='Subscription  Length' , data=df)
plt.title('Subscription  Length by Age')
plt.xlabel('Age')
plt.ylabel('Subscription  Length')
plt.show()
"""

"""
#Boxplot Frequency of use
plt.figure(figsize=(8, 4) , dpi=100)
sns.boxplot(x='Frequency of use' , data=df)
plt.title('Frequency of use')
plt.show()



#Boxplot Frequency of SMS
plt.figure(figsize=(8, 4) , dpi=100)
sns.boxplot(x='Frequency of SMS' , data=df)
plt.title('Frequency of SMS')
plt.show()



#Boxplot Subscription  Length
plt.figure(figsize=(8, 4) , dpi=100)
sns.boxplot(x='Subscription  Length' , data=df)
plt.title('Subscription  Length')
plt.show()



#Histogram Call  Failure
plt.figure(figsize=(8,4) , dpi=100)
sns.histplot(df['Call  Failure'] , bins=10 , color='blue' )
plt.title('Call  Failure')
plt.ylabel('Frequency')
plt.show()



#Histogram Seconds of Use
plt.figure(figsize=(8,4) , dpi=100)
sns.histplot(df['Seconds of Use'] , bins=10 , color='green' , kde=True )
plt.title('Seconds of Use')
plt.ylabel('Frequency')
plt.show()


