from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # برای نزدیک تر کردن بازه های داده های دیتابیس = نرمال سازی
from sklearn.metrics import confusion_matrix # ماتریس درهم ریختگی برای ارزیابی
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score , davies_bouldin_score


#استفاده از متد fetch_usirepo برای لود کردن دیتاست انتخابی با توجه به آیدی گفته شده در سایت مرود نظر
iranian_churn = fetch_ucirepo(id=563)


#استفاده از متد read_csv از پکیج پانداس برای خواندن فایل دیتاست انتخابی
data = pn.read_csv("CustomerChurn.csv")

#print(data.head(4)) برای نمایش 4 تای اول
#print(data.dtypes) برای نمایش ستون ها و نوع داده های آن
#print(data.tail(6)) از آخر نمایش بده




# داده های ناموجود

data1 = data.dropna() #حذف نال ها در سطر ها
data1 = data.dropna(axis=1) #حذف نال ها بصورت در ستون ها
#print(data1)
#print(data.shape)





# آوردن داده های یک ستون مشخص به بازه 0 و 1
mm_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#print(mm_scaler.fit_transform(np.expand_dims(data['Complains'] , axis=1))) تبدیل مقادیر یک ستون به بازه بین 0 و 1



# پیش پردازش 
#  یک مرحله از پیش پردازش که ستون های آورده شده طبق تعریف نباید مقدار صفر داشته باشند ما انها را یافته و بجایشان نال قرار میدهیم سپس با میانگین جایگذاری میکنیم

not_zero = [ 'Age Group' , 'Tariff Plan' , 'Status' , 'Age' ]
for c in not_zero:
    data[c] = data[c].replace(0 , np.NaN)
    mean = int(data[c].mean(skipna=True)) # میانگین با حذف نال هاالبته در پیش پردازش حذف شده
    data[c] = data[c].replace(np.NAN , mean) 
    
    
# حذف سطر های یکسان
data[data.duplicated()==False]
    
#تبدیل داده پیوسته به گسسته
#print(pn.cut(data['Seconds of Use'] , 4))    
    
    
    
    
    
# تقسیم داده به دو قسمت آموزشی و آزمایشی یا همان تست

input = data.drop(['Churn'], axis=1) # بجز ستون تارگت همه موارد ریخته شده اند = ورودی داده
output = data['Churn'] # فقط ستون تارگت = خروجی داده

input_train, input_test, output_train, output_test = train_test_split(input, output, random_state=0 , test_size = 0.3) # 30 درصد برای تست

# نرمال سازی و نزدیک کردن داده ها برای اینکه مدل داده ای رو به دیگر از نظر فراوانی ترجیح و برتری ندهد
# data.describe()
input_train = StandardScaler().fit_transform(input_train) 
input_test = StandardScaler().fit_transform(input_test)




# kmeans

silhouette_scores = []
davies_scores = []

# k_means for diffrent k and random_state=30
# استفاده از این الگوریتم برای داده آموزشی
for n in range(2, 13):
    example = KMeans(n_clusters= n , random_state=30)
    example.fit(input_train)
    data2 = {}
    data2['lables'] = example.labels_
    silhouette = silhouette_score(input_train , data2['lables'])
    davies = davies_bouldin_score(input_train , data2['lables'])
    silhouette_scores.append(silhouette)
    davies_scores.append(davies)



# نمودار ارزیابی نتایج معیار های استفاده شده
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(range(2, 13),silhouette_scores ,  label='silhouette_scores')
plt.title('silhouette_scores')
plt.xlabel('clusters')
plt.ylabel('score')

plt.subplot(1, 2, 2)
plt.plot(range(2, 13), davies_scores, marker='o', label='davies_scores')
plt.title('davies_scores')
plt.xlabel('clusters')
plt.ylabel('score')

plt.legend()
plt.tight_layout()
plt.show()


# بررسی روی داده تست

pre_output = example.predict(input_test)
cluster_counts = Counter(pre_output)
print(cluster_counts)


cmatrix = pn.DataFrame(
    confusion_matrix(output_test , pre_output , labels=[1 , 0]),
    index = ['churn=yes' , 'churn=no'],
    columns = [ 'predict=yes' , 'predict=no']
)
print(cmatrix) 




















