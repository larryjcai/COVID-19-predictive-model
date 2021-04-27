from random import random
#ljc2sh Larry Cai
import numpy as np
import math
import shlex
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
import geopy.distance

#read the file
file = open('VDH-COVID-19-PublicUseDataset-ZIPCode.csv', mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()
X = [0 for i in range(len(lines)-1)]
iter = 0
length = len(lines)
for i in range(1,length):
    l = lines[i].strip().split(',')
    X[i-1] = l
    #X[i-1][0] = datetime.strptime(X[i-1][0], "%m/%d/%Y")

X = np.array(X)
#print(X)
dates = X[:, 0]
dates = [datetime.strptime(d,"%m/%d/%Y") for d in dates]
firstDay = min(dates)
lastDay = max(dates)
#print(X)
unique, counts = np.unique(X[:, 1], return_counts=True)
U = np.asarray((unique, counts)).T
zipData = U[:, 0]
#print(zipData)
zip = zipData[:-2]
#print(zip[0])
#print(zip)
#print(zip)
#print(np.asarray((unique, counts)).T)
#unique zip count =327, which means that there are 327 days
days = int(U[0][1])
#print(days)

file = open('us-zip-code-latitude-and-longitude.csv', mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()

#diction of zipcode to latitude, longitude
zipdict = {}
index = 0
#print(len(lines[1:]))
for line in lines[1:]:
    #zip = latitude, longitude
    l = line.split(';')
    zipdict[int(l[0])] = [l[3], l[4]]
    #zipindex[int(l[0])] = index
    index += 1
    #print(l)

while True:
    try:
        val = str(int(input("Enter a Virginia Zipcode: ")))
        if val not in zip:
            raise ValueError('Not a valid number.')
    except ValueError:
        print('Not a valid number.')
        continue
    else:
        break


while True:
    try:
        predDate = input('Enter the date you want to predict in the format "mm/dd/YYYY":')
        predDate = datetime.strptime(predDate, "%m/%d/%Y")
        if predDate < firstDay:
            print("Before the VDH started recording cases: ")
            raise ValueError("")
    except ValueError:
        print('Not a valid date.')
        continue
    else:
        break

#print(index)
closestZip = np.ones((index, 2))
iter = 0
#print(len(closestZip))
val = int(val)
for zipcode in zipdict:
    if str(zipcode) not in zip:
        continue
    #print(zipcode)
    coords_1 = (zipdict[val][0],zipdict[val][1])
    coords_2 = (zipdict[zipcode][0], zipdict[zipcode][1])
    dist = geopy.distance.distance(coords_1, coords_2).km
    closestZip[iter][0] = zipcode
    closestZip[iter][1] = dist
    iter+=1

closestZip = closestZip[:iter]
#closestZip[:, 0]
sortedZip = closestZip[np.argsort(closestZip[:, 1])]
numZip = 25
closestZip = sortedZip[:numZip]
dict25 = {}
#zipindex = {}
index = 0
#zipindex[int(l[0])] = index
for i in closestZip:
    dict25[i[0]] = index
    index += 1
    #dict25[i[0]] = i[0]
    #print(dict25[i[0]])
    # print(geopy.distance.distance(coords_1, coords_2).km)

#print(dict25[22031])

#read the file
#331 days
totalDays = (lastDay - firstDay).days
xlen = totalDays-30
#last 30 will be my test set
Xtrain = np.ones((xlen,numZip))
Xtest = np.ones((31,numZip))
file = open('VDH-COVID-19-PublicUseDataset-ZIPCode.csv', mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()
X = [0 for i in range(len(lines)-1)]
iter = 0
length = len(lines)
for line in lines[1:]:
    l = line.strip().split(',')
    #print(l[1])
    if l[1] == 'Not Reported':
        continue
    if l[1] == 'Out-of-State':
        continue
    z = int(l[1])
    if z not in dict25:
        continue
    #print(l)
    date = datetime.strptime(l[0], "%m/%d/%Y")
    days = (date - firstDay).days
    #print(l)
    if l[2] == 'Suppressed':
        l[2] = 0
    if l[2] == 'Suppressed*':
        l[2] = 0
    #print(days)
    if days<xlen:
        Xtrain[days][dict25[z]] = l[2]
    else:
        Xtest[days-xlen][dict25[z]] = l[2]

for i in range(len(Xtrain)):
    for j in range(len(Xtrain[0])):
        if Xtrain[i][j] == 0:
            if i>0:
                Xtrain[i][j] = Xtrain[i-1][j]

for i in range(len(Xtest)):
    for j in range(len(Xtest[0])):
        if Xtest[i][j] == 0:
            if i>0:
                Xtest[i][j] = Xtest[i-1][j]

Ytrain = Xtrain[1:, 0]
Xtrain = Xtrain[:-1]
Ytest = Xtest[1:, 0]
Xtest = Xtest[:-1]

#print(Xtrain[0])
#print(Ytest[0])
"""
file = open('city population.csv', mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()

#city by population
cityPop = {}
for line in lines:
    l = line.split(',')
    #print(l)
    cityPop[l[0]] = int(Decimal(l[13].strip()))
    #print(l)
"""
#coords_1 = (zipdict[22031][0],zipdict[22031][1])
#coords_2 = (52.406374, 16.9251681)
"""
Xdate = np.array(X)
Xdate = np.sqeeze(Xdate)
print(Xdate[0])
#print(Xdate[0])
Xsort = Xdate[np.argsort(Xdate[:, 0])]

Xtrain = []
unique, counts = np.unique(Xsort[:, 0], return_counts=True)

print(np.asarray((unique, counts)).T)
"""
#print(Xsort[0])
#Xdate = np.array(X)
#dates = Xdate[:, 0]
#dates = [datetime.strptime(d,"%m/%d/%Y") for d in dates]
#print(dates[0])
#print(min(dates))
#print(max(dates))

#read in the data
#x = pd.read_csv('X.csv', header = None)
#y = pd.read_csv('Y.csv', header = None)
#x = x.to_numpy()
#y = y.to_numpy()
iter = 30

#returns a prediction of the test data

# initialize weights array
"""
weight = [1 for b in range(25)]
test = perceptron(weight, Xtrain, Ytrain, 0.01, iter, Xtest, Ytest)
Xmean = np.mean(Xtrain, axis=0) #,axis=1
Ymean = np.mean(Ytrain, axis=0)
XY = np.matmul(Xtrain,Ytrain)
Xsquare = np.square(Xtrain)
Ysquare = np.square(Ytrain)
#v1 = Xtrain - Xmean
#v2 = Ytrain - Ymean
#v3 = np.square(Xtrain)
#y = ax+b
num = 0
den = 0
for i in range(len(Xtrain)):
    num += (Xtrain[i] - Xmean)*Ytrain[i] - Ymean
    den += np.square(Xtrain[i])
"""
"""
#Y = XB
#B = (X.TX)^(-1)X.T*Y
one = np.array([np.ones(len(Xtrain))]).T
Xtrain = np.concatenate((one, Xtrain), axis=1)

one = np.array([np.ones(len(Xtest))]).T
Xtest = np.concatenate((one, Xtest), axis=1)
#print(len(Xtrain[0]))
#print(Xtest[:, 0])
B = np.matmul(np.matmul(np.matmul(Xtrain.T,Xtrain)**(-1),Xtrain.T),Ytrain)
H = np.matmul(np.matmul(Xtest,np.matmul(Xtest.T,Xtest)**(-1)),Xtest.T)
#print(len(H))
e = np.matmul((np.identity(len(H))-H),Ytest)
Ypred = np.matmul(Xtest,B)
"""
#print(Ytrain)
#print((lastDay - firstDay).days)

from sklearn import linear_model
"""
regr = linear_model.LinearRegression()
lastweek = Ytrain
lastdaycases = lastweek[-1]
firstdaycases = lastweek[-7]
slope = (lastdaycases-firstdaycases)/7
intercept = lastdaycases-(slope*len(lastweek))
#regr.fit(Xtrain, Ytrain)
#Ypred = regr.predict(Xtest)
next30 = range(30)

newInter = Ytrain[-1]
Ypred = newInter+(slope*next30)
strpredDate = str(predDate).split()[0]
print('Number of cases for ' + strpredDate + ' : ', int(newInter+(slope*(predDate-(firstDay + timedelta(days=xlen))).days)))
"""
lastweek = Ytest
lastdaycases = lastweek[-1]
firstdaycases = lastweek[-7]
slope = (lastdaycases-firstdaycases)/7
intercept = lastweek[-1]
next30 = range(30)
Ypred = intercept+(slope*next30)
strpredDate = str(predDate).split()[0]

print('Number of cases for ' + strpredDate + ' : ', int(intercept + (slope * (predDate-timedelta(days=1) - lastDay).days)))
plt.plot(range(iter), Ypred, label='prediction')
#plt.plot(range(iter), Ytest, label='actual')
zipS = str(val)
start = str(lastDay).split()[0]
end = str(lastDay + timedelta(days=30)).split()[0]
plt.title('Number of COVID-19 cases in '+ zipS +' from ' + start + ' to ' + end)
plt.ylabel('Number of COVID-19 casess')
plt.xlabel('Days: from ' + start + ' to ' + end)
plt.legend()
plt.show()

#print(Ytest[0])
#print(Ytest[-1])
#print(Ypred[-1])
#print(Ytrain)

"""

#for i in X:
#zipcode by date
#print(cityPop['Abingdon town']) #=7867

#print(X[0])
#location = geolocator.geocode(X[0][1])
#print((location.latitude, location.longitude))

#Report Date,ZIP Code,Number of Cases,Number of Testing Encounters,Number of PCR Testing Encounters
#print(X[0])
"""

