def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import OneHotEncoder as ohe
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier


def merge_dataset(dt1, dt2, criteria):
	data1 = pd.read_csv(dt1)
	data2 = pd.read_csv(dt2)
	data = data2.merge(data1, on = criteria, how='inner')
	return data

def transform(categorical_columns, numerical_columns, data):
	cat = ('categorical', ohe() , categorical_columns  )
	num = ('numeric', ss(), numerical_columns)
	col_trans = ColumnTransformer([cat, num])
	df_trans_scaled = col_trans.fit_transform(data)
	col_names = get_column_names_from_ColumnTransformer(col_trans)
	for vals in numerical_columns:
		col_names.append(vals)
	df_trans_scaled = pd.DataFrame({col_names[0]: df_trans_scaled[:, 0], col_names[1]: df_trans_scaled[:, 1], col_names[2]: df_trans_scaled[:, 2], col_names[3]: df_trans_scaled[:, 3],col_names[4]: df_trans_scaled[:, 4],col_names[5]: df_trans_scaled[:, 5],col_names[6]: df_trans_scaled[:, 6],col_names[7]: df_trans_scaled[:, 7],col_names[8]: df_trans_scaled[:, 8],col_names[9]: df_trans_scaled[:, 9],col_names[10]: df_trans_scaled[:, 10],col_names[11]: df_trans_scaled[:, 11],col_names[12]: df_trans_scaled[:, 12],col_names[13]: df_trans_scaled[:, 13],col_names[14]: df_trans_scaled[:, 14],col_names[15]: df_trans_scaled[:, 15],col_names[16]: df_trans_scaled[:, 16],col_names[17]: df_trans_scaled[:, 17]})
	return df_trans_scaled, col_names

def get_column_names_from_ColumnTransformer(column_transformer):    
    col_name = []
    for transformer_in_columns in column_transformer.transformers_[:-1]:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1],Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError:
            names = raw_col_name
        if isinstance(names,np.ndarray):
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)
    return col_name

def preprocessing(data, step):
	y_label = data['payment_code']
	yt = y_label
	le = LabelEncoder()
	le.fit(["DEFAULT", "PAYMENT"])
	y_label = le.transform(y_label)
	y_label = pd.DataFrame(y_label)
	sns.countplot(x="payment_code", data=data)
	plt.show()
	data = data.drop('payment_code', axis = 1)
	data = data.drop(['client_id', 'transaction_id', 'contract_id'], axis = 1)
	data['transaction_date'] = pd.to_datetime(data['transaction_date'],unit='s')
	data['transaction_hour'] = data['transaction_date'].dt.hour
	data['transaction_seconds'] = data['transaction_date'].dt.second
	data['transaction_minutes'] = data['transaction_date'].dt.minute
	data['transaction_year'] = data['transaction_date'].dt.year
	data['transaction_month'] = data['transaction_date'].dt.month
	data['transaction_day'] = data['transaction_date'].dt.day
	data = data.drop('transaction_date', axis = 1)
	if step == 0:
		for vals in data:
			if data[vals].dtype == object:
				data = pd.get_dummies(data, columns=[vals])
		col_names = data.columns.tolist()
	else:
		categorical = ['entity_type']
		numeric = ['transaction_hour', 'transaction_seconds', 'transaction_minutes', 'transaction_year', 'transaction_month', 'transaction_day', 'payment_amt', 'entity_year_established']
		data, col_names = transform(categorical, numeric, data)
	return data, y_label, col_names

def split_data(X, Y):
	X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.3, random_state=42)
	return X_Train, Y_Train, X_Test, Y_Test

def pred_decision_Tree(X_Train, Y_Train, X_Test, Y_Test, col_names, make_tree_png = 1):
	X = pd.concat([X_Train, Y_Train], axis=1)
	not_def = X[X[Y_Train.columns.tolist()[0]]==1]
	defa = X[X[Y_Train.columns.tolist()[0]]==0]
	fraud_upsampled = resample(defa,replace=True,n_samples=len(not_def),random_state=27)
	upsampled = pd.concat([not_def, fraud_upsampled])
	class_name = Y_Train.columns.tolist()[0]
	Y_Train = upsampled[Y_Train.columns.tolist()[0]].values
	X_Train = upsampled.drop(class_name, axis=1)
	dtree = DecisionTreeClassifier(criterion = 'gini', random_state=0,min_samples_leaf=2, min_impurity_split = 0.4,max_leaf_nodes=300)
	dtree.fit(X_Train, Y_Train)
	print(f'accuracy Decision tree = {dtree.score(X_Test, Y_Test)}')
	prediction = dtree.predict(X_Test)
	true = []
	for vals in Y_Test.values:
		true.append(vals[0])
	y_actu = pd.Series(true, name='Actual')
	y_pred = pd.Series(prediction, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)
	print(df_confusion)
	f_importance = dtree.feature_importances_
	importances = dtree.feature_importances_
	features = col_names
	indices = np.argsort(importances)
	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), [features[i] for i in indices])
	plt.xlabel('Relative Importance')
	plt.show()
	if make_tree_png == 1:
		dot_data = StringIO()
		export_graphviz(dtree, out_file=dot_data,  
	                filled=True, rounded=True,
	                special_characters=True, feature_names = col_names, class_names=['0','1'])
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
		graph.write_png('tree.png')
		Image(graph.create_png())

def pred_random_forest(X_Train, Y_Train, X_Test, Y_Test, col_names):
	X = pd.concat([X_Train, Y_Train], axis=1)
	not_def = X[X[Y_Train.columns.tolist()[0]]==1]
	defa = X[X[Y_Train.columns.tolist()[0]]==0]
	fraud_upsampled = resample(defa,replace=True,n_samples=len(not_def),random_state=27)
	upsampled = pd.concat([not_def, fraud_upsampled])
	class_name = Y_Train.columns.tolist()[0]
	Y_Train = upsampled[Y_Train.columns.tolist()[0]].values
	X_Train = upsampled.drop(class_name, axis=1)
	dtree = RandomForestClassifier(n_estimators=1000, random_state=0)
	dtree.fit(X_Train, Y_Train)
	f_importance = dtree.feature_importances_
	print(f'accuracy random forest = {dtree.score(X_Test, Y_Test)}')
	prediction = dtree.predict(X_Test)
	true = []
	for vals in Y_Test.values:
		true.append(vals[0])
	y_actu = pd.Series(true, name='Actual')
	y_pred = pd.Series(prediction, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)
	print(df_confusion)
	importances = dtree.feature_importances_
	features = col_names
	indices = np.argsort(importances)
	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), [features[i] for i in indices])
	plt.xlabel('Relative Importance')
	plt.show()




data = merge_dataset("Clients.csv", "Payments.csv", 'client_id')
# X, Y, col_names = preprocessing(data, 1) #uncomment this and comment below line to enable numeric feature transformation
X, Y, col_names = preprocessing(data, 0)
X_Train, Y_Train, X_Test, Y_Test = split_data(X, Y)
pred_decision_Tree(X_Train, Y_Train, X_Test, Y_Test, col_names, 0) 
# pred_decision_Tree(X_Train, Y_Train, X_Test, Y_Test, col_names, 1) #comment the above line and uncomment this line to make a png of decision tree generated
pred_random_forest(X_Train, Y_Train, X_Test, Y_Test, col_names)

