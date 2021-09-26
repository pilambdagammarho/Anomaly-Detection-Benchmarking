import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from commons.constants import Enum
from dataset.dataparser import DataManager
from dataset.modeldataloaders import SiamDataLoader, FeatDataLoader
from models.first_stage import SiameseFeatureExtractor
from models.second_stage import FeatClassifier
from routine.recorder import Record


class Runner():
    """
    Runner class for running the Benchmarking Routine.
    """
    def __init__(self):
        """
        During Constructor call, we first load the configuration from config.yaml.
        This is followed by creating the sampling the signals in the mentioned
        window length.
        We then load the model, wherein the respective dataloaders are created.
        """
        self.config = self.load_config()
        self.DEVICE = self.config["device"]
        self.siamese_config = self.config["siamese"]
        self.class_config = self.config["classifier"]
        self.data = self.load_data()
        # TODO: Load the model and assign it to the feature_extractor
        self.feature_extractor = None
        self.siam, self.classifier = self.load_model()
        if self.siam.training:
            print("INFO: Setting Classifier to train mode since Siamese is in train mode")
            self.classifier.train()

    def load_config(self) -> Dict:
        """
        Reads the config file from desk
        :return: Dict with key:value pairs
        """
        if os.path.isfile(Enum.CONFIG_FILE):
            with open(Enum.CONFIG_FILE, "rb") as file:
                try:
                    configs = yaml.safe_load(file)
                    print(f"INFO: CONFIGURATIONS LOADED!!")
                    return configs
                except yaml.YAMLError as exc:
                    print(exc)
                    exit(-1)
        else:
            print("ERROR: CONFIG FILE IS MISSING!!")

    def run(self) -> None:
        """
        This method iterates over both models. If the model is set at eval mode
        it is sent for inference, if set to training it will be trained and stored at disk.
        """
        for model in [self.siam, self.classifier]:
            if model.training:
                self.epoch(model)
            else:
                self.inference(model)

    def inference(self, model) -> None:
        """
        This method would be invoked to observe the final metric values once the model has been trained.
        The batch size here is hard coded since it does not in any way effect the output.
        :param model: Model to run inference on.
        """
        # Hardcoded batch size for the inference
        _, test_dataloader = self.load_dataloader(model, batch_size=100, feat_extractor=self.feature_extractor)
        loss_fn = nn.BCEWithLogitsLoss(reduce=False)
        total_label, total_output = list(), list()
        total_loss = 0.
        test_record = Record(is_train=False)
        for x, *x_prime, y in test_dataloader:
            model.eval()
            x, y = x.to(self.DEVICE), y.to(self.DEVICE)
            if len(x_prime) > 0:
                x_prime = x_prime[0].to(self.DEVICE)
                output = model(x, x_prime)
            else:
                output = model(x)
            # TODO: Should we include the case weight for the evaluations
            loss = loss_fn(output.squeeze(), y.squeeze())
            total_loss += loss.sum().item()
            output = (torch.sigmoid_(output.squeeze()) > 0.5).float()
            if not y.device.type == "cpu":
                total_output.extend(output.cpu().numpy().ravel())
                total_label.extend(y.cpu().numpy().ravel())
            else:
                total_output.extend(output.numpy().ravel())
                total_label.extend(y.numpy().ravel())
        total_loss /= len(test_dataloader) * test_dataloader.batch_size
        test_record.update(1., total_loss, total_label, total_output)

    def load_epoch_hp(self, model):
        """
        Housekeeping method which reads and loads the basic Hyperparameters
        for the inference or training of the models
        :param model: Based on the type of the model, method automatically reads the said config
        :return:
        """
        loss_dict = {"bcelnr": nn.BCEWithLogitsLoss(reduce=False),
                     "bcel": nn.BCEWithLogitsLoss()}
        optim_dict = {"sgd": torch.optim.SGD,
                      "adam": torch.optim.Adam,
                      "rmsprop": torch.optim.RMSprop}
        if isinstance(model, SiameseFeatureExtractor):
            config_obj = self.siamese_config
        else:
            config_obj = self.class_config
        l_r = config_obj["lr"]
        steps = config_obj["steps"]
        optimizer = optim_dict[config_obj["optim"]]
        batch_size = config_obj["batch"]
        loss_fn = loss_dict[config_obj["loss"]]
        metric = config_obj["metric"]
        eval_interval = config_obj["eval_interval"]
        case_weights = config_obj["case_weights"]
        return steps, l_r, optimizer, batch_size, loss_fn, metric, eval_interval, case_weights

    def load_dataloader(self, model, batch_size, feat_extractor=None):
        """
        This method calls DataManager to load the dataset object which is then followed by creating
        dataloader object for training and evaluation.
        :param model:
        :param batch_size:
        :param feat_extractor:
        :return:
        """
        if isinstance(model, SiameseFeatureExtractor):
            train_dataset = SiamDataLoader(self.data.train_data)
            test_dataset = SiamDataLoader(self.data.test_data)
        if isinstance(model, FeatClassifier):
            train_dataset = FeatDataLoader(self.data.train_data, feat_extractor)
            test_dataset = FeatDataLoader(self.data.test_data, feat_extractor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader

    def epoch(self, model):
        """
        Runs the evaluation and training loop. Evaluation is performed at eval_intervals.
        :param model:
        :return:
        """

        train_record, test_record = Record(is_train=True), Record(is_train=False)
        steps, l_r, optimizer, batch_size, loss_fn, metric, eval_interval, case_weights = self.load_epoch_hp(model)
        train_dataloader, test_dataloader = self.load_dataloader(model, batch_size, self.feature_extractor)

        optimizer = optimizer(model.parameters(), lr=l_r)
        for epoch in range(steps):
            # run inference or evaluation
            total_loss = 0.
            if epoch % eval_interval == 0:
                total_output, total_label = list(), list()
                with torch.no_grad():
                    for x, *x_prime, y in test_dataloader:
                        model.eval()
                        x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                        if len(x_prime) > 0:
                            x_prime = x_prime[0].to(self.DEVICE)
                            output = model(x, x_prime)
                        else:
                            output = model(x)
                        loss = loss_fn(output.squeeze(), y.squeeze())
                        total_loss += loss.mean().item()
                        output = (torch.sigmoid_(output.squeeze()) > 0.5).float()
                        if not y.device.type == "cpu":
                            total_output.extend(output.cpu().numpy().ravel())
                            total_label.extend(y.cpu().numpy().ravel())
                        else:
                            total_output.extend(output.numpy().ravel())
                            total_label.extend(y.numpy().ravel())
                    total_loss /= len(test_dataloader) * test_dataloader.batch_size
                    test_record.update(epoch, total_loss, total_label, total_output)
                    # Handle the loss here

            # perform the training loop
            total_output, total_label = list(), list()
            for x, *x_prime, y in train_dataloader:
                model.train()
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                if len(x_prime) > 0:
                    x_prime = x_prime[0].to(self.DEVICE)
                    output = model(x, x_prime)
                else:
                    output = model(x)
                loss = loss_fn(output.squeeze(), y.squeeze())
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                output = (torch.sigmoid_(output.squeeze()) > 0.5).float()
                if not y.device.type == "cpu":
                    total_output.extend(output.cpu().numpy().ravel())
                    total_label.extend(y.cpu().numpy().ravel())
                else:
                    total_output.extend(output.numpy().ravel())
                    total_label.extend(y.numpy().ravel())
            total_loss /= len(test_dataloader) * test_dataloader.batch_size
            train_record.update(epoch, total_loss, total_label, total_output)
        # assign the model for further computations
        if self.feature_extractor is None:
            path = f"siamese_{self.data.signal_name}_{l_r}_{batch_size}_{str(test_record.get_lowest_loss())}.pt"
            self.feature_extractor = model.eval()
        else:
            path = f"classifier_{self.data.signal_name}_{l_r}_{batch_size}_{str(test_record.get_lowest_loss())}.pt"
        path = os.path.join("./models", "stored_model", path)
        torch.save(model.state_dict(), path)

        # perform plotting of the values
        train_record.plot_values(f"{model.name} TRAINING {self.data.data_name} LR:{l_r} Batch Size:{batch_size}")
        test_record.plot_values(f"{model.name} TESTING {self.data.data_name} LR:{l_r} Batch Size:{batch_size}")
        test_record.plot_losses(train_record.loss, f"{model.name} Loss {self.data.data_name} LR:{l_r} Batch Size:{batch_size}")

    def load_data(self) -> DataManager:
        """
        Method that interacts with reading the dataset from disk and sampling the signal
        in the defined windows
        :return: DataManger Object
        """
        # obtain the values here
        data_config = self.config["data"]
        data_name = data_config["name"]
        window = data_config["window"]
        signal_name = data_config["signal"]
        anomaly_ratio = data_config["anomaly_ratio"]
        t_t_split = data_config["train_test_split"]
        datamanager = DataManager(data_name, window,
                                  anomaly_ratio, t_t_split, signal_name)
        return datamanager

    def load_model(self) -> Tuple[SiameseFeatureExtractor, FeatClassifier]:
        """
        This method loads the model. It decides if the train parameter is set to false to read the weights from
        the disk and set the model to eval mode. Else load the model and set it to train mode.
        :return: Tuple of Models
        """
        def obtain_model(model, config):
            if config["train"]:
                model.train().double()
            else:
                model_path = config["model_name"]
                model_path = os.path.join("./models", "stored_model", model_path)
                if os.path.isfile(model_path):
                    # TODO: Load the weights to the model
                    model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
                    model.double().eval()
                    if model.name == "Siamese":
                        self.feature_extractor = model
                    print("INFO: Loaded weights for the Siamese Model")
                else:
                    print(f"ERROR: Model path {model_path} for siamese cannot be found")
                    exit(-1)
            return model

        siamese_layers = self.siamese_config["layers"]
        siamese_activation = self.siamese_config["activation"]
        feature_dimensions = self.siamese_config["feature_dimensions"]
        siamese_train_mode = self.siamese_config["train"]
        window_length = self.data.window
        siamese = SiameseFeatureExtractor(siamese_layers,
                                          self.data.data_name,
                                          feature_dimensions,
                                          siamese_activation,
                                          window_length).to(self.DEVICE)
        siamese = obtain_model(siamese, self.siamese_config)

        class_hidden_layer = self.class_config["hidden_layer"]
        class_activation = self.class_config["activation"]
        class_train_mode = self.class_config["train"]
        if siamese.training:
            self.class_config["train"] = True
        classifer = FeatClassifier(feature_dimensions, class_hidden_layer, class_activation).to(self.DEVICE)
        classifer = obtain_model(classifer, self.class_config)

        print(f"INFO: Siamese model loaded as\n {siamese}\n")
        print(f"INFO: Classifier model loaded as\n {classifer}\n")
        return siamese, classifer


if __name__ == "__main__":
    Runner().run()
