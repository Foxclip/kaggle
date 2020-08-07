import pandas as pd
import sys
sys.path.insert(0, "e:\\Projects\\Python\\nn_sim\\")
import nn_sim.dataset as dataset

if __name__ == "__main__":

    # Increasing number of columns so all of them are showed
    pd.set_option('display.max_columns', 5)

    # loading datasets
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    combined_df = pd.concat([train_df, test_df])

    target_col = "SalePrice"

    # transforming dataset

    df = combined_df.copy()

    # dropping unneeded columns
    # df = dataset.drop(df, ["Name", "PassengerId", "Ticket", "Cabin"])
    df = df[[
        "LotArea",
        "YearBuilt",
        "SalePrice"
    ]]

    # filling missing values
    # df = dataset.impute(df, ["Fare", "Embarked"])

    # df = dataset.label_encode(df, [])
    # df = dataset.one_hot_encode(df, [
    #     "Sex", "Embarked", "Pclass", "Age", "Fare"
    # ])
    df, scalers = dataset.scale(df)

    # print(df)
    # import sys
    # sys.exit(0)

    # specifying settings of a model
    model_settings = dataset.NeuralNetworkSettings()
    model_settings.task_type = dataset.TaskTypes.regression
    model_settings.output_count = 1
    model_settings.epochs = 400
    model_settings.folds = 10
    model_settings.target_col = target_col
    model_settings.validation = dataset.ValidationTypes.val_split
    model_settings.checkpoint = False
    model_settings.gpu = False

    # specifying lists of parameters
    layers_lst = [1]
    neurons_lst = [3]
    # layers_lst = [2, 3, 4, 5]
    # neurons_lst = [16, 32, 64, 128, 256, 512]

    # loading and preparing data
    X_train, X_test = dataset.cut_dataset(df, target_col=target_col)

    # training models and saving file with predictions on test dataset
    dataset.train_models(X_train, model_settings, layers_lst, neurons_lst)

    # making predictions with the best model
    predict = dataset.make_predictions(X_test, scalers)
    final_df = pd.DataFrame(test_df["Id"])
    final_df[target_col] = predict
    final_df.to_csv("output.csv", index=False)
