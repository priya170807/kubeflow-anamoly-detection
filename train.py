
# Import python libraries
import json
import os
import numpy as np
import pandas as pd
import pickle
from io import StringIO
import config
# Import sklearn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
# Import tensorflow modules
import tensorflow as tf
from tensorflow.python.lib.io import file_io 


class isolation_forest():
    """
    declare the class variables
    """
    
    def __init__(self):
    

        self.input_df = pd.read_csv(config.processed_data).drop(['Unnamed: 0'],axis=1)
        # self.input_df = pd.read_csv(filepath)
        print("/n")
        print("the input dataframe")
        print(self.input_df.head())
        self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.split_size = self.preprocess(self.input_df)
        self.model = self.train_isolation_forest()
        self.yhat, self.acc, self.specificity, self.sensitivity = self.score_isolation_forest()
        # self.yhat, self.acc = self.score_isolation_forest()
        self.dump_artifacts()

    def preprocess(self, input_data):
        """
        Preprocesses the data and splits the data into train and test for 
        training and prediction.
        """
        df = input_data.copy()
        df = df.drop("Time", axis=1)
        X = df.loc[:, df.columns != 'Class']
        X["Amount"] = MinMaxScaler().fit_transform(X["Amount"].values.reshape(-1,1))
        X = X # converts the df to a numpy array
        y = df['Class']
        split_size = int(0.8 * df.shape[0])
        X_train = X[:split_size]
        X_test = X[split_size:]
        y_train = y[:split_size]
        y_test = y[split_size:]
        return X_train, X_test, y_train, y_test, split_size


    def train_isolation_forest(self):
        """
        Train the isolation-forest model
        """
        print(f"The length of Xtrain={len(self.Xtrain)}")
        print(self.Xtrain.head())
        anamoly_df = self.ytrain[self.ytrain.values==1]
        print(f"The length of anamoly_df={len(anamoly_df)}")
        outliers_fraction = round(len(anamoly_df)/len(self.Xtrain), 5)
        print(f"outliers_fraction = {outliers_fraction}")
        outliers_frac = outliers_fraction * 10
        model =  IsolationForest(contamination=outliers_frac)
        model.fit(self.Xtrain.values) 
        # save the model to disk    
        pickle.dump(model, open("./isolation_forest_model", 'wb'))
        return model

    
    def score_isolation_forest(self):
        """
        The function scores the Xtest data and creates a new dataframe
        results.
        """
        results_isolation_forest_df = pd.DataFrame()
        results_isolation_forest_df["Time"] = self.input_df["Time"][self.split_size:]
        results_isolation_forest_df["Amount"] = self.input_df["Amount"][self.split_size:]
        results_isolation_forest_df["Transaction_type"] =  self.input_df["Class"][self.split_size:]
        results_isolation_forest_df["anamoly_scores"] = self.model.predict(self.Xtest.values)
        results_isolation_forest_df = results_isolation_forest_df.reset_index(drop=True)
        print(f"The length of results_isolation_forest_df = {len(results_isolation_forest_df)}")
        print("The first five rows of results_isolation_forest_df")
        print("\n")
        print(results_isolation_forest_df.head())
        results_isolation_forest_df["anamoly_scores"].replace(-1, 0, inplace=True)
        # print(unique_val)
        yhat = results_isolation_forest_df["anamoly_scores"].values
        y_actual = self.ytest.values
        acc = np.mean(yhat==y_actual)
        print("/n")
        print("the accuracy", acc)
        print("/n")
        print("The y_actual values")
        print(y_actual[:5])
        print("The y_test values")
        print(yhat[:5])
        tn, fp, fn, tp = confusion_matrix(y_actual, yhat).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp + fn)

        # # Now print to file
        with tf.io.gfile.GFile(config.store_artifacts + "/metrics.json", 'w') as outfile:
            json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)
        return yhat, acc, specificity, sensitivity
       
    # Saving Model Artifacts like confusion matrix values and other metrics

    def dump_artifacts(self):
        
        print("Inside dump_artifacts")
        vocab = list(np.unique(self.ytest.values))
        print("vocab values")
        print(vocab)
        cm = confusion_matrix(self.ytest.values, self.yhat, labels=vocab)

        data = []
        for target_index, target_row in enumerate(cm):
            for predicted_index, count in enumerate(target_row):
                data.append((vocab[target_index], vocab[predicted_index], count))

        df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
        df_cm['target'] = df_cm['target'].astype(str)
        df_cm['predicted'] = df_cm['predicted'].astype(str)
        replacements = {
            'target': {
                str(0) : 'non-fraud',
                str(1) : 'fraud' 
            },
            'predicted' : {
                str(0) : 'non-fraud',
                str(1) : 'fraud'
            }
        }
        df_cm.replace(replacements, inplace=True)
        cm_file = os.path.join(config.store_artifacts, 'confusion_matrix.csv')
        # cm_file = os.path.join('confusion_matrix.csv')
        print("the path of cm_file", cm_file)
        with file_io.FileIO(cm_file, 'w') as f:
            df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

        metadata = {
            'outputs' : [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {"name": "target", "type": "CATEGORY"},
                {"name": "predicted", "type": "CATEGORY"},
                {"name": "count", "type": "NUMBER"},
            ],
            'source': cm_file,
            # Convert vocab to string because for bealean values we want "True|False" to match csv data.
            'labels': list(map(str, vocab)),
            }
            ]
        }
        with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
            json.dump(metadata, f)

        metrics = {
            'metrics': [{
            'name': 'accuracy-score',
            'numberValue':  self.acc,
            'format': "PERCENTAGE",
            }, 
            {
            'name': 'specificity',
            'numberValue':  self.specificity,
            'format': "PERCENTAGE",
            }, 
            {
            'name': 'sensitivity',
            'numberValue':  self.sensitivity,
            'format': "PERCENTAGE",
            },]
        }
        with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
            json.dump(metrics, f)


if __name__ == '__main__':
    obj = isolation_forest()
    # obj = isolation_forest('./data_processed.csv')


