import pandas as pd
import sys
sys.path.insert(0, "e:\\Projects\\Python\\nn_sim\\")
import nn_sim.dataset as dataset
import nn_sim.simulation as simulation

if __name__ == "__main__":

    # Increasing number of columns so all of them are showed
    pd.set_option('display.max_columns', 5)

    # loading datasets
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    dataset.load_dataset(combined_df)
    dataset.target_col = "SalePrice"

    # transforming dataset

    # dropping unneeded columns
    # df = dataset.drop(df, ["Name", "PassengerId", "Ticket", "Cabin"])
    dataset.leave_columns([
        "MSSubClass",
        "MSZoning",
        "LotFrontage",
        "LotArea",
        "Street",
        "Alley",
        "LotShape",
        "YearBuilt",
        "YearRemodAdd",
        "SalePrice",
    ])

    # df = dataset.swap(df, ["LotFrontage"], "NA", None)

    # filling missing values
    dataset.impute([
        "LotFrontage",
    ])
    dataset.fillna([
        "Alley"
    ])

    # df = dataset.label_encode(df, [])
    dataset.one_hot_encode([
        "MSSubClass",
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
    ])
    scalers = dataset.scale()

    # df.to_csv("df.csv")
    # print(df)
    # import sys
    # sys.exit(0)

    # specifying settings of a model
    model_settings = simulation.NeuralNetworkSettings()
    model_settings.task_type = simulation.TaskTypes.regression
    model_settings.output_count = 1
    model_settings.epochs = 40
    model_settings.folds = 10
    model_settings.bin_count = 3
    model_settings.validation = simulation.ValidationTypes.cross_val
    model_settings.unscale_loss = True
    model_settings.checkpoint = False
    model_settings.gpu = False

    # specifying lists of parameters
    # layers_lst = [1]
    # neurons_lst = [3]
    layers_lst = [2, 3, 4, 5]
    neurons_lst = [16, 32, 64, 128, 256, 512]

    # training models and saving file with predictions on test dataset
    dataset.train_models(model_settings, layers_lst, neurons_lst)

    # making predictions with the best model
    predict = dataset.make_predictions()
    final_df = pd.DataFrame(test_df["Id"])
    final_df[dataset.target_col] = predict
    final_df.to_csv("output.csv", index=False)
