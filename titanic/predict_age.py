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
    df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    # transforming dataset (feature engineering, encoding, etc.)
    df = dataset.drop(df, [
        "Sex", "Name", "PassengerId", "Ticket", "Cabin", "Embarked",
        "Pclass", "SibSp", "Parch", "Survived"
    ])
    df = dataset.impute(df, ["Fare"])
    df = dataset.scale(df, exclude_cols="Age")

    # print(df)
    # df.to_csv("age.csv")
    # import sys
    # sys.exit(0)

    # specifying settings of a model
    model_settings = dataset.NeuralNetworkSettings()
    model_settings.task_type = dataset.TaskTypes.regression
    model_settings.intermediate_activations = "relu"
    model_settings.output_count = 1
    model_settings.optimizer = "Adam"
    model_settings.batch_size = 32
    model_settings.epochs = 320

    # specifying lists of parameters
    layers_lst = [1]
    neurons_lst = [3]

    # loading and preparing data
    data_split, X_test = dataset.split_data(df, target_col="Age")

    # training models and saving file with predictions on test dataset
    dataset.train_models(data_split, model_settings, layers_lst, neurons_lst)

    # making predictions with the best model
    predict = dataset.make_predictions(X_test)

    # write them to new dataframe
    df = pd.read_csv("train.csv")
    predict_df = pd.DataFrame(predict)
    predict_df.index = X_test.index
    predict_col = predict_df.iloc[:, 0]
    df["Age"] = df["Age"].fillna(predict_col)
    df.to_csv("train_age.csv", index=False)
