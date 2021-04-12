# flake8: noqa
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace


# Split the dataframe into test and train data
def split_data(df):
    a = np.empty(0, dtype=str)
    columns = np.append(a, np.arange(0, 128))
    y = df['cluster']
    X = df[columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=8)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data, class_args):
    class_model = DecisionTreeClassifier(**class_args)
    class_model.fit(data["train"]["X"], data["train"]["y"])
    return class_model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}
    return metrics


def create_sample_data_csv(
    aml_workspace: Workspace,
    datastore_name: str,
    file_name: str = "COVID19Articles.csv",
    for_scoring: bool = False):

    url = \
        "https://solliancepublicdata.blob.core.windows.net" + \
        "/ai-in-a-day/lab-02/"
    df = pd.read_csv(url + file_name)
    if for_scoring:
        df = df.drop(columns=['cluster'])
    df.to_csv(file_name, index=False)
    
    if (datastore_name):
      datastore = Datastore.get(aml_workspace, datastore_name)
    else:
       datastore = Datastore.get_default(aml_workspace)
    datastore.upload_files(
        files=[file_name],
        overwrite=True,
        show_progress=False,
    )

def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str = "COVID19Articles.csv",
) -> Dataset:
    if (datastore_name):
      datastore = Datastore.get(aml_workspace, datastore_name)
    else:
       datastore = Datastore.get_default(aml_workspace)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset


def main():
    print("Running train_aml.py")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="COVID19Articles_model_github.pkl",
    )

    parser.add_argument(
        "--step_output",
        type=str,
        help=("output for passing data to next step")
    )

    parser.add_argument(
        "--dataset_version",
        type=str,
        help=("dataset version")
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,\
               a new version of the dataset will be registered")
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help=("Dataset name. Dataset must be passed by name\
              to always get the desired dataset version\
              rather than the one used while the pipeline creation")
    )

    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [step_output]: %s" % args.step_output)
    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [dataset_name]: %s" % args.dataset_name)
    
    datastore_name = os.environ.get("DATASTORE_NAME")
    model_name = args.model_name
    step_output_path = args.step_output
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()

    # Get the dataset
    if (dataset_name):
        if (data_file_path == ""):
            if (dataset_name in Dataset.get_all(run.experiment.workspace).keys()):
              dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, version=dataset_version)  
            else:
              create_sample_data_csv(run.experiment.workspace, datastore_name)
              dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       datastore_name)
        else:
            dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       datastore_name,
                                       data_file_path)
    else:
        if (data_file_path == ""):
            data_file_path = "COVID19Articles.csv"
            create_sample_data_csv(run.experiment.workspace, datastore_name)
        dataset_name = "COVID19Articles_Training_githubactions"
        dataset = register_dataset(run.experiment.workspace,
                                  dataset_name,
                                  datastore_name,
                                  data_file_path)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset

    # Split the data into test/train
    df = dataset.to_pandas_dataframe()
    data = split_data(df)

    class_args = {"max_depth": 5}
    # Train the model
    model = train_model(data, class_args)

    # Evaluate and log the metrics returned from the train function
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        run.log(k, v)
        
    # files saved in the "outputs" folder are automatically uploaded into run history
    model_file_name = "COVID19Articles_model.pkl"
    joblib.dump(model, os.path.join('outputs', model_file_name))
    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    run.complete()

if __name__ == '__main__':
    main()
