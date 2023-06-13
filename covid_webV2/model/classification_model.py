import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import torch.nn as nn

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def create_class_model(model_name, in_channels, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "densenet":
        model_ft = models.densenet201()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

class ClassificationModel(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_class_model(model_name, model_hparams["in_channels"], 
                                  model_hparams["num_classes"], False, use_pretrained=True)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        
    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)
    
    def shared_step(self, batch, stage):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
                
        return {
            "loss": loss,
            "labels": labels,
            "preds": preds
        }
    
    def shared_epoch_end(self, outputs, stage):
        labels = torch.cat([x["labels"] for x in outputs]).cpu()
        preds = torch.cat([x["preds"] for x in outputs]).cpu().argmax(dim=-1)
        
        acc = (preds == labels).float().mean()        
        
        metrics = {
             f"{stage}_acc": acc,            
         }
        
        self.log_dict(metrics, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            return optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)