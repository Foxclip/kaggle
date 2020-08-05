import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "e:\\Projects\\Python\\nn_sim\\")
import nn_sim.dataset as dataset

if __name__ == "__main__":

    # Increasing number of columns so all of them are showed
    pd.set_option('display.max_columns', 20)

    # loading datasets
    test_df = pd.read_csv("test.csv")
    combined_df = pd.read_csv("train_age.csv")

    # transforming dataset

    df = combined_df.copy()

    # dropping unneeded columns
    df = dataset.drop(df, ["Name", "PassengerId", "Ticket", "Cabin"])

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

    # # name prefixes
    # def func(name):
    #     parts = name.split(",")
    #     prefix = parts[1].split(" ")[1]
    #     return prefix
    #
    # df["Name"] = df["Name"].map(func)

    df = dataset.label_encode(df, [])
    # df = one_hot_encode(df, ["Sex", "Embarked", "Pclass"])
    df = dataset.one_hot_encode(df, [
        "Sex", "Embarked", "Pclass", "Age", "Fare"
    ])
    df = dataset.scale(df, exclude_cols=["Survived"])

    # print(df)
    # import sys
    # sys.exit(0)

    # specifying settings of a model
    model_settings = dataset.NeuralNetworkSettings()
    model_settings.task_type = dataset.TaskTypes.binary_classification
    model_settings.output_count = 1
    model_settings.epochs = 4
    model_settings.folds = 10
    model_settings.target_col = "Survived"
    model_settings.validation = dataset.ValidationTypes.none

    # specifying lists of parameters
    layers_lst = [5]
    neurons_lst = [512]
    # layers_lst = [2, 3, 4, 5]
    # neurons_lst = [16, 32, 64, 128, 256, 512]

    # loading and preparing data
    X_train, X_test = dataset.cut_dataset(df, target_col="Survived")

    # training models and saving file with predictions on test dataset
    dataset.train_models(X_train, model_settings, layers_lst, neurons_lst)

    # making predictions with the best model
    predict = np.round(dataset.make_predictions(X_test))
    final_df = pd.DataFrame(test_df["PassengerId"])
    final_df["Survived"] = predict
    final_df["Survived"] = final_df["Survived"].astype(int)
    final_df.to_csv("output.csv", index=False)
