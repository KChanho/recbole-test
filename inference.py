from recbole.utils import get_trainer
from recbole.quick_start import load_data_and_model


if __name__ == '__main__':

    # model loading and initialization
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='saved/BPR-Jan-12-2023_02-26-49.pth',
    )

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)