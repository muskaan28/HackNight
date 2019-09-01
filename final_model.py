# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 02:40:17 2019

@author: Ashu
"""

from azureml.core import Workspace
#ws = Workspace.create(
#            name='learn-workspace-py',
#            subscription_id='a55f7f68-d7e1-462b-92b0-787f71d7f77c', 
#            resource_group='Hacknight',
#            create_resource_group=False,
#            location='eastus2'
#)




from pyrebase import pyrebase
import pandas as pd

config = {
    "apiKey": "AIzaSyBCmoC7fFVKRsztfHL_HqmQo29ooLvRBpw",
    "authDomain": "hacknight-65e59.firebaseapp.com",
    "databaseURL": "https://hacknight-65e59.firebaseio.com",
    "projectId": "hacknight-65e59",
    "storageBucket": "hacknight-65e59.appspot.com",
    "messagingSenderId": "582741754484",
    "appId": "1:582741754484:web:801a11d057c78a8e"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

res = db.child("employee").get().val().values()

leaveApplications = list() 
for _tuple in res:
    for _data in _tuple.values():
        leaveApplications.append(_data)

training = pd.DataFrame(leaveApplications)
training.to_csv(r"C:\Users\Ashu\Downloads\training.csv")
training = pd.read_csv("C:\\Users\\Ashu\\Downloads\\training.csv")

training['Gender'] = training['Gender'].replace(['M','male','m','maile','Male-ish','Cis Male','something kinda male?','Mal','Make','Guy (-ish) ^_^','Man','male leaning androgynous','Malr','msle','Mail','cis male','Male ','ostensibly male, unsure what that really means','Cis Man'],'Male')
training['Gender'] = training['Gender'].replace(['F','female','f','femail','Cis Female','Woman','cis-female/femme','Female (trans)','Female (cis)','femake','woman','Female ','Femake','femail',],'Female')
training['Gender'] = training['Gender'].replace(['Trans-female','p', 'non-binary','Nah', 'Enby', 'fluid', 'Genderqueer','Androgyne', 'Agender', 'Trans woman', 'Neuter', 'queer','A little about you'],'Others')

training['Gender'] = training['Gender'].replace(["Male"],0)
training['Gender'] = training['Gender'].replace(["Female"],1)
training['Gender'] = training['Gender'].replace(["Others"],2)

training['remote_work'] = training['remote_work'].replace(["No"],0)
training['remote_work'] = training['remote_work'].replace(["Yes"],1)


training['tech_company'] = training['tech_company'].replace(["No"],0)
training['tech_company'] = training['tech_company'].replace(["Yes"],1)

training['benefits'] = training['benefits'].replace(["No"],0)
training['benefits'] = training['benefits'].replace(["Yes"],1)
training['benefits'] = training['benefits'].replace(["Don't know"],2)

training['care_options'] = training['care_options'].replace(["No"],0)
training['care_options'] = training['care_options'].replace(["Yes"],1)
training['care_options'] = training['care_options'].replace(["Not sure"],2)


training['wellness_program'] = training['wellness_program'].replace(["No"],0)
training['wellness_program'] = training['wellness_program'].replace(["Yes"],1)
training['wellness_program'] = training['wellness_program'].replace(["Don't know"],2)


training['seek_help'] = training['seek_help'].replace(["No"],0)
training['seek_help'] = training['seek_help'].replace(["Yes"],1)
training['seek_help'] = training['seek_help'].replace(["Don't know"],2)

training['anonymity'] = training['anonymity'].replace(["No"],0)
training['anonymity'] = training['anonymity'].replace(["Yes"],1)
training['anonymity'] = training['anonymity'].replace(["Don't know"],2)

training['leave'] = training['leave'].replace(["Somewhat difficult", "Very difficult"],0)
training['leave'] = training['leave'].replace(["Somewhat easy", "Very easy"],1)
training['leave'] = training['leave'].replace(["Don't know"],2)

training['mental_health_consequence'] = training['mental_health_consequence'].replace(["No"],0)
training['mental_health_consequence'] = training['mental_health_consequence'].replace(["Yes"],1)
training['mental_health_consequence'] = training['mental_health_consequence'].replace(["Maybe"],2)

training['phys_health_consequence'] = training['phys_health_consequence'].replace(["No"],0)
training['phys_health_consequence'] = training['phys_health_consequence'].replace(["Yes"],1)
training['phys_health_consequence'] = training['phys_health_consequence'].replace(["Maybe"],2)

training['coworkers'] = training['coworkers'].replace(["No"],0)
training['coworkers'] = training['coworkers'].replace(["Yes"],1)
training['coworkers'] = training['coworkers'].replace(["Some of them"],2)

training['supervisor'] = training['supervisor'].replace(["No"],0)
training['supervisor'] = training['supervisor'].replace(["Yes"],1)
training['supervisor'] = training['supervisor'].replace(["Some of them"],2)

training['mental_health_interview'] = training['mental_health_interview'].replace(["No"],0)
training['mental_health_interview'] = training['mental_health_interview'].replace(["Yes"],1)
training['mental_health_interview'] = training['mental_health_interview'].replace(["Maybe"],2)

training['phys_health_interview'] = training['phys_health_interview'].replace(["No"],0)
training['phys_health_interview'] = training['phys_health_interview'].replace(["Yes"],1)
training['phys_health_interview'] = training['phys_health_interview'].replace(["Maybe"],2)

training['mental_vs_physical'] = training['mental_vs_physical'].replace(["No"],0)
training['mental_vs_physical'] = training['mental_vs_physical'].replace(["Yes"],1)
training['mental_vs_physical'] = training['mental_vs_physical'].replace(["Don't know"],2)

training['self_employed'] = training['self_employed'].replace(["No"],0)
training['self_employed'] = training['self_employed'].replace(["Yes"],1)
training['self_employed'] = training['self_employed'].replace(["Don't know"],2)
training.self_employed.fillna(0,inplace=True)

training['obs_consequence'] = training['obs_consequence'].replace(["No"],0)
training['obs_consequence'] = training['obs_consequence'].replace(["Yes"],1)

training.family_history.unique()
training['family_history'] = training['family_history'].replace(["No"],0)
training['family_history'] = training['family_history'].replace(["Yes"],1)

training['work_interfere'] = training['work_interfere'].replace(["Never"],0)
training['work_interfere'] = training['work_interfere'].replace(["Rarely"],1)
training['work_interfere'] = training['work_interfere'].replace(["Sometimes"],2)
training['work_interfere'] = training['work_interfere'].replace(["Often"],3)
training.work_interfere.fillna(2, inplace = True)

import numpy as np
training.loc[training.Age > 100, 'Age'] = np.nan
training.loc[training.Age < 18, 'Age'] = np.nan
ages = training[(training['Age']<=100 ) & (training['Age']>=18)]
median = ages.median()
training.fillna(median,inplace=True)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

training.apply(pd.to_numeric, errors="ignore")

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

p = training["empid"]
training = training.drop(columns = ["Unnamed: 0", "empid"])
#training.sort_index(axis = 1, inplace = True)
training.to_csv(r"C:\Users\Ashu\Downloads\testing_cleaned.csv")

X_train = pd.read_csv("C:\\Users\\Ashu\Desktop\\X_train.csv")
X_train.sort_index(axis = 1, inplace = True)
X_train = X_train.drop(columns = ["Unnamed: 0"])
X_train.to_csv(r"C:\Users\Ashu\Downloads\training_cleaned.csv")
X_test = pd.read_csv("C:\\Users\\Ashu\\Downloads\\testing_cleaned.csv")
X_test = X_test.drop(columns = ["Unnamed: 0"])
y_train = pd.read_csv("C:\\Users\\Ashu\\Downloads\\trainms.csv")
y_train["treatment"] = y_train["treatment"].replace(["No"],0)
y_train["treatment"] = y_train["treatment"].replace(["Yes"],1)
y_train = y_train.iloc[:,8:9]


from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
svc=SVC(probability=True, kernel='linear')
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
result={}
for i in range(len(p)):
    emp = p[i]
    result["empid"]=emp
    if y_pred[i]:
        result["treatment"]="Yes"
    else:
        result["treatment"]="No"
    
    db.child("hr").child(emp).push(result)
#y_pred = pd.DataFrame(y_pred)
###print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#
############################################################################
## Code for machine learning and prediction
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.model_selection import train_test_split
#from sklearn import metrics
#import pandas as pd
#
#import numpy as np
#import pandas as pd
#import seaborn as sns
#sns.set_palette('husl')
#import matplotlib.pyplot as plt
#%matplotlib inline
#
#from sklearn import metrics
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split

#training = pd.read_csv("C:\\Users\\Ashu\\Downloads\\trainms1.csv")
    
adaBoostClassifier_filename = "adaBoostClassifier.pkl"
adaBoostClassifier_pkl = open(adaBoostClassifier_filename, 'wb')
import pickle
pickle.dump(model, adaBoostClassifier_pkl)
pickle.dump(model, adaBoostClassifier_pkl)
adaBoostClassifier_pkl.close()