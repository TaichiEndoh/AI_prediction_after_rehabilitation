import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecs
import os
from sklearn.model_selection import train_test_split

class my_o_directory:
    def __init__(self,pass_out):
        self.pass_out = pass_out
    def print_name(self):
        print(self.pass_out)
    def pass_o(self):
        return self.pass_out
    def pass_out_new(self):
        data_dir = self.pass_out
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(u.pass_o()+"Folder created")
    def imp_data(self):
        image_file_path = './simulated_rehabilitation_patient_data.csv'
        def open_rehabilitation_data(image_file_path_v):
            with codecs.open(image_file_path_v, "r", "Shift-JIS", "ignore") as file:
                    dfpp = pd.read_table(file, delimiter=",")    
            dfpp_m_rehabilitation = dfpp
            return dfpp_m_rehabilitation
               
        rehabilitation1=open_rehabilitation_data(image_file_path)
        return rehabilitation1

#define the value of self here
u = my_o_directory("./output/")
u.print_name()
u.pass_out_new()
Read_data=u.imp_data()

all_data_p1=Read_data
all_data_after_get_dummies_gender=pd.get_dummies(all_data_p1['gender'])
all_data_after_get_dummies_gender.reset_index()
all_data_non_gender=all_data_p1.drop(["gender"], axis=1)
all_data_o = pd.concat([all_data_after_get_dummies_gender, all_data_non_gender], axis=1)
all_data_out=all_data_o.fillna(0)
all_data_out.describe()
merge_data=all_data_out.rename(columns={"F1":"target"})

#Separate test data from train data
def make_test_vol_train(merge_data):
    # Isolate the objective variable
    X = merge_data.drop("target",axis=1).values
    y = merge_data["target"].values
    columns_name = merge_data.drop("target",axis=1).columns
    def Test_data_and_training_data_split(df,X,Y):
             N_train = int(len(df) * 0.86)
             N_test = len(df) - N_train
             X_train, X_test, y_train, y_test = \
                train_test_split(X, Y, test_size=N_test,shuffle=False,random_state=42)
             return X_train, X_test, y_train, y_test
    # Execute a function that separates data for training and data for testing.
    X_train, X_test, y_train, y_test = Test_data_and_training_data_split(merge_data,X,y)
    X_train = pd.DataFrame(X_train, columns=columns_name)
    X_test = pd.DataFrame(X_test, columns=columns_name)
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)
    test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
    test_df=test_dfp.rename(columns={0:"target"})
    y_trainp = pd.DataFrame(y_train)
    X_trainp = pd.DataFrame(X_train)
    train=pd.concat([y_trainp, X_trainp], axis=1)
    merge_data_p=train.rename(columns={0:"target"})
    X = merge_data_p.drop("target",axis=1).values
    y = merge_data_p["target"].values
    columns_name = merge_data_p.drop("target",axis=1).columns
    def Test_data_and_training_data_split(df,X,Y):
             N_train = int(len(df) * 0.80)
             N_test = len(df) - N_train
             X_train, X_test, y_train, y_test = \
                train_test_split(X, Y, test_size=N_test,random_state=42)
             return X_train, X_test, y_train, y_test
    # Execute a function that separates the data for training from the data for validation.
    X_train,X_val, y_train,y_val = Test_data_and_training_data_split(merge_data_p,X,y)
    X_train = pd.DataFrame(X_train, columns=columns_name)
    X_val = pd.DataFrame(X_val, columns=columns_name)
    #training verification Combine test data vertically
    y_trainp = pd.DataFrame(y_train)
    X_trainp = pd.DataFrame(X_train)
    train=pd.concat([y_trainp, X_trainp], axis=1)
    y_valp = pd.DataFrame(y_val)
    X_valp = pd.DataFrame(X_val)
    val=pd.concat([y_valp, X_valp], axis=1)
    train_vol=pd.concat([train, val])
    order_of_things=train_vol.rename(columns={0:"target"})
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)
    test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
    test_df=test_dfp.rename(columns={0:"target"})
    marge_data_out=pd.concat([order_of_things, test_df])
    return marge_data_out

marge_data_out = make_test_vol_train(merge_data)
#Save in CSV format
marge_data_out.to_csv(r''+u.pass_o()+"Essay_output_data_first_F1_target.csv", encoding = 'shift-jis')
#Read CSV
with codecs.open(u.pass_o()+"Essay_output_data_first_F1_target.csv", "r", "Shift-JIS", "ignore") as file:
    F1_target_data_pre = pd.read_table(file, delimiter=",")    
F1_target_data = F1_target_data_pre.drop(['Unnamed: 0'], axis=1)
F1rename_target_data=F1_target_data.rename(columns={"target":"F1"})
first_num="F1"
y_variable=F1rename_target_data[first_num]
feature_value=F1rename_target_data.drop(['F1','F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
       'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'], axis=1)
y_x_variable_pre=pd.concat([y_variable, feature_value], axis=1)
y_x_variable_date=y_x_variable_pre.rename(columns={first_num:"target",'女性':"woman",'男性':"man"})
y_x_variable_non_b=y_x_variable_date.drop(["Admission Day"], axis=1)
#5 or less and 6 or more to binarry data.
df_b1=y_x_variable_non_b.astype('int')
df_b1['flg']=df_b1['target'].where(df_b1['target'] > 5, 0)
df_b2=df_b1
df_b2['flg']=df_b1['flg'].replace([6,7],[1,1])
y_x_variable_flg=df_b2.drop(["target"], axis=1)
y_x_variable=y_x_variable_flg.rename(columns={'flg': 'target'})
y_x_variable

#Predicted by pycaret
#Predict F1 at discharge as an objective variable
from pycaret.classification import *
clf1 = setup(y_x_variable, target ='target',train_size = 0.86,data_split_shuffle=False,fold=10,silent=True,session_id=42)
best_model = compare_models()
