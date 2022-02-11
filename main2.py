import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("datasets/car_state.csv")
print(data.head());
print(data.iloc[-5:])
print(data.describe())


fig, axs=plt.subplots(3,2)

axs[0,0].scatter(data["buying_price"],data["status"],10, c='red', marker='o', alpha=0.7)
axs[0,0].set_xlabel("buying price", color="red")
axs[0,0].set_ylabel("status", color="black")

axs[1,0].scatter(data["maintenance"],data["status"],10, c='blue', marker='o', alpha=0.7)
axs[1,0].set_xlabel("maintenance", color="blue")
axs[1,0].set_ylabel("status", color="black")


axs[2,0].scatter(data["doors"],data["status"],10, c='green', marker='o', alpha=0.7)
axs[2,0].set_xlabel("doors", color="green")
axs[2,0].set_ylabel("status", color="black")

axs[0,1].scatter(data["seats"],data["status"],10, c='pink', marker='o', alpha=0.7)
axs[0,1].set_xlabel("seats", color="pink")
axs[0,1].set_ylabel("status", color="black")

axs[1,1].scatter(data["trunk_size"],data["status"],10, c='purple', marker='o', alpha=0.7)
axs[1,1].set_xlabel("trunk_size", color="purple")
axs[1,1].set_ylabel("status", color="black")


axs[2,1].scatter(data["safety"],data["status"],10, c='orange', marker='o', alpha=0.7)
axs[2,1].set_xlabel("safety", color="orange")
axs[2,1].set_ylabel("status", color="black")


def retType(type):
    if type=="medium":
        return 0.5
    elif type=="low":
        return 0.25
    elif type=='very high':
        return 1
    elif type=="high":
        return 0.75

def getNumber(num):
    if num=="5 or more":
        return (5-min)/(5-min)
    else:
        return (float(num)-min)/(5-min)

def getSize(size):
    if size=="small":
        return 0.165
    elif size=="medium":
        return 0.495
    elif size=="big":
        return 0.825

min=float(min(data["doors"].values))

trasformacijaFeatures=pd.DataFrame(data=[[retType(type)for type in data["buying_price"]],[retType(type)for type in data["maintenance"]],\
                                 [getNumber(num)for num in data["doors"]], [getNumber(num)for num in data["seats"]],\
                                 [getSize(num)for num in data["trunk_size"]],[retType(num)for num in data["safety"]] ])

def transformValue(value):
    if value=="unacceptable":
        return 0
    else:
        return 1

trasformacijaTarget=pd.DataFrame( data=[transformValue(value) for value in data["status"]])

trasformacijaFeatures=trasformacijaFeatures.T
duzina=trasformacijaFeatures.loc[0,].size




X = data.iloc[:, :5]
y = data.iloc[:, 6]




def calculate_distance(point1, point2):
    nDimension=6
    sum=0
    while nDimension>0:
        sum+=pow(point1[7-nDimension] - point2[7-nDimension],2)
        nDimension=nDimension-1
    return math.sqrt(sum)

# sum = 0;
# for(int i=0;i<N;++i){
#     sum += pow(p[i]-q[i],2);
# sqrt

def find_closest_distance(TestSet, point, n):
    distances=[]
    index1=[]
    for value in TestSet.itertuples():
        distances.append(calculate_distance(value,point))
        index1.append(value[0])
    dataFrameDist=pd.DataFrame(data=distances, index=index1, columns=['dist'])

    return dataFrameDist.sort_values(by=['dist'], axis=0)[:n]





def knn_prediction(featuresTrain,featuresTest, targetTrain, targetTest, n):
    from collections import Counter

    output=[]
    for point in featuresTest.itertuples():
        nDistances = find_closest_distance(featuresTrain, point,n)
        elem=[]
        for i in nDistances.index:
            elem.append(targetTrain.loc[i,0])
        counter=Counter( elem)
        output.append(counter.most_common()[0][0])

    return output




featuresTrainSet, featuresTestSet, targetTrainSet, targetTestSet = train_test_split(trasformacijaFeatures,trasformacijaTarget, test_size=0.25, random_state=1)

resault=knn_prediction(featuresTrainSet, featuresTestSet, targetTrainSet, targetTestSet, 3)

print(f"Tacnost mog modela {accuracy_score(targetTestSet,resault)}")

model=KNeighborsClassifier(n_neighbors=3)
model.fit(trasformacijaFeatures,trasformacijaTarget.values.ravel())

buying_price=0.49
maintenance=0.49
doors=0.33
seats=0.66
trunk_size=0.165
safety=0.49

example_estate=pd.DataFrame(data=[[buying_price],[maintenance], [doors],[seats],[trunk_size],[safety]])
example_estate=example_estate.T

predicted= model.predict(featuresTestSet)


print(f'Tacnost ugradjenog modela {accuracy_score(targetTestSet,predicted)}')



plt.tight_layout()
plt.show()