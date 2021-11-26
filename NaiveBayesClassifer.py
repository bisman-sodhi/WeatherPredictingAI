import numpy as np
import pandas as pd
import time
#import matplotlib.pyplot as plt
import sys


#data = pd.read_csv("training.txt")
#data.head(22)
def editTrainFile(trainData):
    startDF = time.time()
    df = pd.read_csv(trainData , sep = ',', names=["Location", "MinTemp", "MaxTemp", "Rainfall",
    "Evaporation", "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", 
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm", "RainToday", "RainTomorrow"])

    endDF = time.time()
    #print("Took " , endDF - startDF , " sec to read \n") #0.017827749252319336 
    X = df.drop([df.columns[-1]], axis = 1)
    #X1 = df[df.columns[1]]
    #X2 = df.columns[1]
    #print("X1",X1) #prints all rows in column 1
    #print("X2", X2) #prints only the name of column 1
    #X3 = df[1]
    #print("X3 ", X3) #doesn't work
    y = df[df.columns[-1]]
    
    
    del df["MaxTemp"]
    del df["Temp3pm"]
    del df["Location"]
    del df["MinTemp"]
    del df["Temp9am"]
    
    #df = df.drop(df["Location"])
    df["RainToday"] = df["RainToday"].astype('category')
    df["RainToday"] = df["RainToday"].cat.codes
    df["RainToday"] = df["RainToday"].astype('float64')

    df["RainTomorrow"] = df["RainTomorrow"].astype('category')
    df["RainTomorrow"] = df["RainTomorrow"].cat.codes
    df["RainTomorrow"] = df["RainTomorrow"].astype('float64')
    y = y.astype('category') 
    y = y.cat.codes
    y = y.astype('float64')

    #Giving numbers to the 16 directions
    df["WindGustDir"] = df["WindGustDir"].astype('category')
    df["WindGustDir"] = df["WindGustDir"].cat.codes
    df["WindGustDir"] = df["WindGustDir"].astype('float64')

    #Giving numbers to the 16 directions
    df["WindDir9am"] = df["WindDir9am"].astype('category')
    df["WindDir9am"] = df["WindDir9am"].cat.codes
    df["WindDir9am"] = df["WindDir9am"].astype('float64')

    #Giving numbers to the 16 directions
    df["WindDir3pm"] = df["WindDir3pm"].astype('category')
    df["WindDir3pm"] = df["WindDir3pm"].cat.codes
    df["WindDir3pm"] = df["WindDir3pm"].astype('float64')
    '''
    #Giving numbers to the 26 Locations
    df["Location"] = df["Location"].astype('category')
    df["Location"] = df["Location"].cat.codes
    df["Location"] = df["Location"].astype('float64')
    '''
    return df

def editTestFile(testData):
    startTest = time.time()

    test = pd.read_csv(testData , sep = ',', names=["Location", "MinTemp", "MaxTemp", "Rainfall",
    "Evaporation", "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", 
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm", "RainToday", "RainTomorrow"])
    endTest = time.time()

    #print("Took ", endTest - startTest, " sec to read test \n") #0.0029497146606445312

    X = test.drop([test.columns[-1]], axis = 1)
    y = test[test.columns[-1]]
    #test = test.drop(df["Location"])
    
    
    del test["MaxTemp"]
    del test["Temp3pm"]
    del test["Location"]
    del test["MinTemp"]
    del test["Temp9am"]
    
    test["RainToday"] = test["RainToday"].astype('category')
    test["RainToday"] = test["RainToday"].cat.codes
    test["RainToday"] = test["RainToday"].astype('float64')

    test["RainTomorrow"] = test["RainTomorrow"].astype('category')
    test["RainTomorrow"] = test["RainTomorrow"].cat.codes
    test["RainTomorrow"] = test["RainTomorrow"].astype('float64')
    y = y.astype('category') 
    y = y.cat.codes
    y = y.astype('float64')

    #Giving numbers to the 16 directions
    test["WindGustDir"] = test["WindGustDir"].astype('category')
    test["WindGustDir"] = test["WindGustDir"].cat.codes
    test["WindGustDir"] = test["WindGustDir"].astype('float64')

    #Giving numbers to the 16 directions
    test["WindDir9am"] = test["WindDir9am"].astype('category')
    test["WindDir9am"] = test["WindDir9am"].cat.codes
    test["WindDir9am"] = test["WindDir9am"].astype('float64')

    #Giving numbers to the 16 directions
    test["WindDir3pm"] = test["WindDir3pm"].astype('category')
    test["WindDir3pm"] = test["WindDir3pm"].cat.codes
    test["WindDir3pm"] = test["WindDir3pm"].astype('float64')
    '''
    #Giving numbers to the 26 Locations
    test["Location"] = test["Location"].astype('category')
    test["Location"] = test["Location"].cat.codes
    test["Location"] = test["Location"].astype('float64')
    '''
    return test
    
def outlier(dataset, feature):
    #use quantile function to get the first and third quadrant
    firstQ = dataset[feature].quantile(0.25)
    thirdQ = dataset[feature].quantile(0.75)
    #formaula
    difference = thirdQ - firstQ
    upperBound = thirdQ + 1.5 * difference
    lowerBound = firstQ - 1.5 * difference
    listOutlier = dataset.index[ (dataset[feature] < lowerBound) | (dataset[feature] > upperBound) ]  #index of outlier
    return listOutlier

def removeOutlier(dataset, ls):
    ls = sorted(set(ls))
    dataset = dataset.drop(ls)
    return dataset

def correlation(dataset, threshold):
    col_cor = set()
    corr_matrix = dataset.corr()
    for item in range(len(corr_matrix.columns)):
        for secItem in range(item):
            if (corr_matrix.iloc[item,secItem]) > threshold:
                colname = corr_matrix.columns[item]
                col_cor.add(colname)
    return col_cor

def prior(dataset, attributeName):
    classes = np.unique(dataset[attributeName])
    prior = []
    for i in classes:
        prior.append(len(dataset.groupby(attributeName).get_group(i))/len(dataset))
    return prior

def cacl(x, mean, std):
    #pdf formula
    return (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((x-mean)**2 / (2 * std**2 )))

def llhood(dataset, attributeName, attributeData, className, classLabel):
    #dataset.groupby(column name).get_group(distinct values in column)
    dataset  = dataset.groupby(className).get_group(classLabel)
    #mean and std of column
    mean= dataset[attributeName].mean()
    std = dataset[attributeName].std()
    likelihood = cacl(attributeData, mean, std)
    #return natural log to avoid dealing with really small numbers
    return np.log(likelihood)

def gnb(dataset, attribute, className):
    #store outcome in this
    outcome = []
    #get all column names
    features = list(dataset.columns)[:-1]
    #Prior prob of P(RainTomorrow = Yes) P(rainTomorrow = No)
    Theprior = prior(dataset, className)
    
    for x in attribute:
       # print("x ", x )
        labels = np.unique(dataset[className]) 
        #print("labesl ", labels)
        likelihood = [1]*len(labels)
        #print(len(likelihood))
        #for item in (len(labels)):   doesn't work
        #for every yes/no
        for item in range(len(labels)):
            #for secItem in len(features):
            #for every column given yes. for every column given no:
            for secItem in range(len(features)):
                #likelihood[Yes] += llhood()
                #likelihood[No] += llhood()
                likelihood[item] += (llhood(dataset, features[secItem], x[secItem], className, labels[item]))
        #postProb = P(x | c)P(c)
        postProb = [1]*len(labels)
        for item in range(len(labels)):
            postProb[item] = likelihood[item] + np.log(Theprior[item]) 
        outcome.append(np.argmax(postProb))

    return np.array(outcome)

def acc(inTest, myPred):
    counter = 0
    for i in range(len(inTest)):
        if inTest[i] == myPred[i]:
            counter += 1
    return (counter/float(len(inTest))) * 100

def main():
    trainData = sys.argv[1]
    testData = sys.argv[2]
    df = editTrainFile(trainData)
    test = editTestFile(testData)
    
    newList = []
    newTestList = []
    '''
    remcol = correlation(df, 0.90)
    for x in remcol:
        del df[x]
        del test[x]
    '''
    for col in df:
        newList.extend(outlier(df, col))
    #outlier(df)
    for col in test:
        newTestList.extend(outlier(test, col))
    #newTestList = newList
    df = removeOutlier(df, newList)
    test = removeOutlier(test, newTestList)
    #print("newList" , newList)
    
    X_test = test.iloc[:,:-1].values
    Y_test = test.iloc[:,-1].values
    X = df.iloc[:, :-1].values
    y = df.iloc[:,-1].values
    startGNB = time.time()
    yPred = gnb(df, X_test, className="RainTomorrow")
    endGNB = time.time()
    #print("time to classify training data ", endGNB - startGNB, "\n")
    
    for i in yPred:
        print(int(i))
    ans = acc(Y_test, yPred)
    print("ans is ", ans)
main()

