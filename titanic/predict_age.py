import pandas as pd
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

    # transforming dataset (feature engineering, encoding, etc.)
    df = combined_df.copy()

    # dropping unneeded columns
    df = dataset.drop(df, [
        "PassengerId", "Ticket", "Cabin", "Survived"
    ])

    # filling missing data
    df = dataset.impute(df, ["Fare", "Embarked"])

    # family member count
    df["Family"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = df["Family"] == 0

    # name prefixes
    def func(name):
        parts = name.split(",")
        prefix = parts[1].split(" ")[1]
        return prefix

    df["Name"] = df["Name"].map(func)

    # fare categories
    # df["Fare"] = pd.cut(
    #     df["Fare"],
    #     (0, 10, 100, 600),
    #     include_lowest=True,
    #     labels=["0-10", "10-100", "100-600"],
    # )

    df = dataset.one_hot_encode(df, [
        "Sex", "Embarked", "Pclass", "Name"
    ])
    df = dataset.scale(df, exclude_cols="Age")

    # df.to_csv("age.csv")
    # import sys
    # sys.exit(0)

    # specifying settings of a model
    model_settings = dataset.NeuralNetworkSettings()
    model_settings.task_type = dataset.TaskTypes.regression
    model_settings.intermediate_activations = "relu"
    model_settings.epochs = 25
    model_settings.target_col = "Age"
    model_settings.validation = dataset.ValidationTypes.cross_val
    model_settings.folds = 10

    # specifying lists of parameters
    # layers_lst = [2, 3]
    # neurons_lst = [2, 4, 8, 16, 32, 64, 128]
    layers_lst = [3, 4, 5]
    neurons_lst = [4, 8, 16, 32, 64, 128, 256, 512]

    # loading and preparing data
    X_train, X_test = dataset.cut_dataset(df, target_col="Age")

    # training models and saving file with predictions on test dataset
    dataset.train_models(X_train, model_settings, layers_lst, neurons_lst)

    # making predictions with the best model
    predict = dataset.make_predictions(X_test)

    # write them to new dataframe
    predict_df = pd.DataFrame(predict)
    predict_df.index = X_test.index
    predict_col = predict_df.iloc[:, 0]
    final_df = combined_df.copy()
    final_df["Age"] = final_df["Age"].fillna(predict_col)
    final_df.to_csv("train_age.csv", index=False)
