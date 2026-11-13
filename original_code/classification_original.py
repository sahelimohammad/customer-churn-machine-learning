from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # برای نزدیک تر کردن بازه های داده های دیتابیس = نرمال سازی
from sklearn.neighbors import KNeighborsClassifier # پیاده سازی درخت KNN
from sklearn.metrics import accuracy_score # هر چهار تای ارزیابی ها اینجا هست f1_score , Precision , recall
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier #  وارد کردن درخت تصمیم
from sklearn.metrics import confusion_matrix # ماتریس درهم ریختگی برای ارزیابی
from sklearn.model_selection import cross_val_score



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



# KNN K=7
k = KNeighborsClassifier(n_neighbors=7) # ساخت مدل که حفظ میکنه
k.fit(input_train,output_train) # آموزش دادن مدل به کمک داده های آموزشی
pre_output = k.predict(input_test)# پیش بینی کردن با داده های تست
#print(accuracy_score(output_test , pre_output)) # محاسبه میزان دقت با خروجی واقعی و خروجی پیش بینی شده
print('k=7', '\n' , classification_report(output_test , pre_output),'\n') # محاسبه recall , f1-score , precision
#print(confusion_matrix(output_test , pre_output)) # محاسبه ماتریس درهم ریختگی
cmatrix = pn.DataFrame(
    confusion_matrix(output_test , pre_output , labels=[1 , 0]),
    index = ['churn=yes' , 'churn=no'],
    columns = [ 'predict=yes' , 'predict=no']
)
print(cmatrix)



print('=======================================================================\n')

#KNN K=13
k = KNeighborsClassifier(n_neighbors=13) # ساخت مدل که حفظ میکنه
k.fit(input_train,output_train) # آموزش دادن مدل به کمک داده های آموزشی
pre_output = k.predict(input_test)# پیش بینی کردن با داده های تست
#print(accuracy_score(output_test , pre_output)) # محاسبه میزان دقت با خروجی واقعی و خروجی پیش بینی شده
print('k=13', '\n' ,classification_report(output_test , pre_output),'\n') # محاسبه recall , f1-score , precision
#print(confusion_matrix(output_test , pre_output)) # محاسبه ماتریس درهم ریختگی
cmatrix = pn.DataFrame(
    confusion_matrix(output_test , pre_output , labels=[1 , 0]),
    index = ['churn=yes' , 'churn=no'],
    columns = [ 'predict=yes' , 'predict=no']
)
print(cmatrix)


print('=======================================================================\n')


# Decision Tree k=3
D = DecisionTreeClassifier(max_depth=3) # ساخت مدل
D.fit(input_train,output_train) # آموزش دادن مدل به کمک داده های آموزشی
pre_output = D.predict(input_test)# پیش بینی کردن با داده های تست
#print(accuracy_score(output_test , pre_output)) # محاسبه میزان دقت با خروجی واقعی و خروجی پیش بینی شده
print('depth=3', '\n' ,classification_report(output_test , pre_output)) # محاسبه recall , f1-score , precision
#print(confusion_matrix(output_test , pre_output),'\n') # محاسبه ماتریس درهم ریختگی

cmatrix = pn.DataFrame(
    confusion_matrix(output_test , pre_output , labels=[1 , 0]),
    index = ['churn=yes' , 'churn=no'],
    columns = [ 'predict=yes' , 'predict=no']
)
print(cmatrix) # نمایش ماتریس درهم ریختگی بصورت ساده و مفهومی

print('=======================================================================\n')




# Decision Tree k=8
D = DecisionTreeClassifier(max_depth=8) # ساخت مدل
D.fit(input_train,output_train) # آموزش دادن مدل به کمک داده های آموزشی
pre_output = D.predict(input_test)# پیش بینی کردن با داده های تست
#print(accuracy_score(output_test , pre_output)) # محاسبه میزان دقت با خروجی واقعی و خروجی پیش بینی شده
print('depth=8', '\n' ,classification_report(output_test , pre_output)) # محاسبه recall , f1-score , precision
#print(confusion_matrix(output_test , pre_output),'\n') # محاسبه ماتریس درهم ریختگی

cmatrix = pn.DataFrame(
    confusion_matrix(output_test , pre_output , labels=[1 , 0]),
    index = ['churn=yes' , 'churn=no'],
    columns = [ 'predict=yes' , 'predict=no']
)
print(cmatrix) # نمایش ماتریس درهم ریختگی بصورت ساده و مفهومی

print('=======================================================================\n')






# برای کراس ولیدیشن دوباره کا ان ان و درخت تصمیم را با 5 انجام میدهیم

#KNN K=5
k = KNeighborsClassifier(n_neighbors=5) # ساخت مدل که حفظ میکنه
k.fit(input_train,output_train) # آموزش دادن مدل به کمک داده های آموزشی
pre_output = k.predict(input_test)# پیش بینی کردن با داده های تست
#print(accuracy_score(output_test , pre_output)) # محاسبه میزان دقت با خروجی واقعی و خروجی پیش بینی شده
print('k=5', '\n' ,classification_report(output_test , pre_output),'\n') # محاسبه recall , f1-score , precision
#print(confusion_matrix(output_test , pre_output)) # محاسبه ماتریس درهم ریختگی
cmatrix = pn.DataFrame(
    confusion_matrix(output_test , pre_output , labels=[1 , 0]),
    index = ['churn=yes' , 'churn=no'],
    columns = [ 'predict=yes' , 'predict=no']
)
print(cmatrix)


print('=======================================================================\n')


# Decision Tree
D = DecisionTreeClassifier(max_depth=5) # ساخت مدل
D.fit(input_train,output_train) # آموزش دادن مدل به کمک داده های آموزشی
pre_output = D.predict(input_test)# پیش بینی کردن با داده های تست
#print(accuracy_score(output_test , pre_output)) # محاسبه میزان دقت با خروجی واقعی و خروجی پیش بینی شده
print('depth=5', '\n' ,classification_report(output_test , pre_output)) # محاسبه recall , f1-score , precision
#print(confusion_matrix(output_test , pre_output),'\n') # محاسبه ماتریس درهم ریختگی

cmatrix = pn.DataFrame(
    confusion_matrix(output_test , pre_output , labels=[1 , 0]),
    index = ['churn=yes' , 'churn=no'],
    columns = [ 'predict=yes' , 'predict=no']
)
print(cmatrix) # نمایش ماتریس درهم ریختگی بصورت ساده و مفهومی


print('=======================================================================\n')


# K-Fold Cross Validation K=cv=5 مقایسه با KNN
cross = cross_val_score(estimator=k , X=input_train , y=output_train , cv=10)
print("Accuracy: {:.2f} %".format(cross.mean()*100) , 'with KNN')


print('=======================================================================\n')


# K-Fold Cross Validation K=cv=5 مقایسه با درخت تصمیم
cross = cross_val_score(estimator=D , X=input_train , y=output_train , cv=10)

print("Accuracy: {:.2f} %".format(cross.mean()*100) , 'with Decision tree')




