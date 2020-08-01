import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "e:\\Projects\\Python\\nn_sim\\")
import nn_sim.dataset as dataset

if __name__ == "__main__":

    # Increasing number of columns so all of them are showed
    pd.set_option('display.max_columns', 20)

    # loading datasets
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    # transforming dataset

    df = combined_df.copy()

    # filling missing values
    df = dataset.impute(df, ["Fare", "Embarked"])

    # whether passenger is alone
    df["Family"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = df["Family"] == 0
    df = dataset.drop(df, ["SibSp", "Parch", "Family"])

    # age categories
    df["Age"] = pd.cut(
        df["Age"],
        (0, 18, 35, 60, 120),
        labels=["Child", "Young", "Middle", "Old"]
    )
    # fare categories
    df["Fare"] = pd.cut(
        df["Fare"],
        (0, 10, 100, 600),
        include_lowest=True,
        labels=["0-10", "10-100", "100-600"],
    )

    df = dataset.label_encode(df, [])
    # df = one_hot_encode(df, ["Sex", "Embarked", "Pclass"])
    df = dataset.one_hot_encode(df, [
        "Sex", "Embarked", "Pclass", "Age", "Fare"
    ])
    df = dataset.drop(df, ["Name", "PassengerId", "Ticket", "Cabin"])
    df = dataset.scale(df, exclude_cols=["Survived"])

    # print(df)
    # import sys
    # sys.exit(0)

    # specifying settings of a model
    model_settings = dataset.NeuralNetworkSettings()
    model_settings.task_type = dataset.TaskTypes.binary_classification
    model_settings.intermediate_activations = "relu"
    model_settings.output_count = 1
    model_settings.optimizer = "Adam"
    model_settings.batch_size = 32
    model_settings.epochs = 320

    # specifying lists of parameters
    # layers_lst = [1, 2, 3]
    # neurons_lst = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    layers_lst = [1]
    neurons_lst = [3]

    # loading and preparing data
    data_split, X_test = dataset.split_data(df, target_col="Survived")

    # training models and saving file with predictions on test dataset
    dataset.train_models(data_split, model_settings, layers_lst, neurons_lst)

    # making predictions with the best model
    predict = np.round(dataset.make_predictions(X_test))
