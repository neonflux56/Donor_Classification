
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf



if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='test.csv')
    #parser.add_argument('--features', type=str, default = (pd.read_csv(args.train_file).columns[:-1]) )  # in this script we ask user to explicitly name features
    #parser.add_argument('--target', type=str, default = (pd.read_csv(args.train_file).columns[-1])) # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    print('reading data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    features = train_df.columns[:-1]
    target = train_df.columns[-1]
    
    print('building training and testing datasets')
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[target]
    y_test = test_df[target]

    # train
    print('training model')
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1)
    
    model.fit(X_train, y_train)

    print('validating model')
    y_pred = model.predict(X_test)
    results = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(results)

    print("Classification Report:")
    target_names = ['Lower Salary class', 'Higher Salary class']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # print couple perf metrics
    #for q in [10, 50, 90]:
    #    print('AE-at-' + str(q) + 'th-percentile: '
    #          + str(np.percentile(a=abs_err, q=q)))
        
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('model persisted at ' + path)
    print(args.min_samples_leaf)

    
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf