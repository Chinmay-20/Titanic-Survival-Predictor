import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
	#step 1 : Load data
	titanic_data=pd.read_csv("Titanic.csv")
	
	print("first 5 entries from loaded dataset")
	print(titanic_data.head())
	
	print("Number of passengers are "+str(len(titanic_data)))
	
	#step 2 : analyze data
	
	print("visualization : survived and non survived passengers")
	plt.figure()
	target="Survived"
	countplot(data=titanic_data,x=target).set_title("Survived and Non Survived passengers")
	plt.show()
	
	print("visualization : survived and non survived passengers based on Gender")
	plt.figure()
	target="Survived"
	countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and non survived passengers based on gender")
	plt.show()
	
	print("visualization : survived and non survived passengers on Passenger class")
	plt.figure()
	target="Survived"
	countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and non survived passengers based on Passenger class")
	plt.show()
	
	print("visualization : survived and non survived passengers on Age")
	plt.figure()
	titanic_data["Age"].plot.hist().set_title("Survived and Non Survived passengers based on Age")
	plt.show()
	
	print("visualization : survived and non survived passengers on Fare")
	plt.figure()
	titanic_data["Fare"].plot.hist().set_title("Survived and Non Survived passengers based on Fare")
	plt.show()
	
	#step 3 : data cleaning
	
	titanic_data.drop("zero",axis=1,inplace=True)
	
	print("First 5 entries from loaded dataset after removing zero column")
	print(titanic_data.head(5))
	
	print("Values of sex column")
	print(pd.get_dummies(titanic_data["Sex"]))
	
	print("values of sex column after removing one field")
	Sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
	print(Sex.head(5))
	
	print("Values of Pclass column after removing one field")
	Pclass=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
	print(Pclass.head(5))
	
	print("Values of dataset after contenating new columns")
	titanic_data=pd.concat([titanic_data,Sex,Pclass],axis=1)
	print(titanic_data.head(5))
	
	print("Values of data set after removing irrelevant columns")
	titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
	print(titanic_data.head(5))
	
	x=titanic_data.drop("Survived",axis=1)
	y=titanic_data["Survived"]
	
	#Step 4 : data training
	xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)
	
	logmodel=LogisticRegression()
	
	logmodel.fit(xtrain,ytrain)
	
	#step 5 : data testing
	
	prediction=logmodel.predict(xtest)
	
	#step 6 : calculate accuracy
	
	print("Classification report of Logistic regression is : ")
	print(classification_report(ytest,prediction))
	
	print("confusion matrix of logistic regression is : ")
	print(confusion_matrix(ytest,prediction))
	
	print("Accuracy of Logistic Regression is : ")
	print(accuracy_score(ytest,prediction))
	
def main():
	print("Supervised Machine Learning")
	
	print("logistic regression on titanic dataset")
	
	TitanicLogistic()

if __name__=="__main__":
	main()
