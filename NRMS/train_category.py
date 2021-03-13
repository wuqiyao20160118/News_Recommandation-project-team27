import torch
from torch.utils import data
import pytorch_lightning as pl
import pdb

from models.deep_cross import DeepCrossWithCategory
from dataset.NewsDatasetCategory import NewsDatasetCategory, NewsDatasetCategoryVal
from utils.utils import ndcg_score, mrr_score


class NRMSCrossCategoryModel(pl.LightningModule):
    def __init__(self, params):
        super(NRMSCrossCategoryModel, self).__init__()
        self.hyperParams = params
        self.model = DeepCrossWithCategory(params)

    def prepare_data(self):
        """
        inherit from pytorch lightning module
        :return:
        """
        train_news_dataset = NewsDatasetCategory(self.hyperParams, self.hyperParams["train_data_path"])
        val_news_dataset = NewsDatasetCategoryVal(self.hyperParams, self.hyperParams["val_data_path"])
        self.train_data, _ = data.random_split(train_news_dataset, [int(len(train_news_dataset)*0.99),
                                                                    len(train_news_dataset)-int(len(train_news_dataset)*0.99)])
        self.val_data, _ = data.random_split(val_news_dataset, [int(len(val_news_dataset) * 0.95),
                                                                    len(val_news_dataset) - int(
                                                                        len(val_news_dataset) * 0.95)])
        self.test_data = NewsDatasetCategoryVal(self.hyperParams, self.hyperParams["val_data_path"])

    def train_dataloader(self):
        """
        inherit from pytorch lightning module, assembling train dataloader
        :return:
        """
        train_loader = data.DataLoader(self.train_data, num_workers=self.hyperParams["num_workers"],
                                       batch_size=self.hyperParams["batch_size"], shuffle=self.hyperParams["shuffle"])
        return train_loader

    def val_dataloader(self):
        """
        inherit from pytorch lightning module, assembling validation dataloader
        :return:
        """
        val_loader = data.DataLoader(self.val_data, num_workers=self.hyperParams["num_workers"],
                                     batch_size=1, shuffle=self.hyperParams["shuffle"])
        return val_loader

    def test_dataloader(self):
        """
        inherit from pytorch lightning module, assembling test dataloader
        :return:
        """
        test_loader = data.DataLoader(self.test_data, num_workers=self.hyperParams["num_workers"],
                                      batch_size=1, shuffle=self.hyperParams["shuffle"])
        return test_loader

    def configure_optimizers(
            self,
    ):
        """
        inherit from pytorch lightning module, configuring the optimizer
        :return: optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperParams['lr'], weight_decay=1e-5)
        return optimizer

    def forward(self):
        return None

    def training_step(self, batch, batch_idx):
        """
        inherit from pytorch lightning module, implements a mini-batch training step
        :param batch: batch data
        :param batch_idx: batch index
        :return: dictionary containing loss (prediction can also be contained)
        """
        clicks_title, clicks_abstract, candidates_title, candidates_abstract, clicks_category_feature, labels = batch
        loss = self.model(clicks_title, clicks_abstract, candidates_title, candidates_abstract, clicks_category_feature, labels)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """
        inherit from pytorch lightning module, implements after each epoch ends
        :param outputs: List of outputs you defined in training_step
        :return: dictionary containing statistics
        """
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.model.eval()
        logs = {'train_loss': mean_loss}
        self.log_dict(logs, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """
        inherit from pytorch lightning module, implements a mini-batch validation step
        :param batch: batch data
        :param batch_idx: batch index
        :return: dictionary containing evaluation metrics on training step
        """
        clicks_title, clicks_abstract, candidates_title, candidates_abstract, clicks_category_feature, labels = batch
        with torch.no_grad():
            activation = self.model(clicks_title, clicks_abstract, candidates_title, candidates_abstract, clicks_category_feature)
        mrr = 0.0
        auc = 0.0
        ndcg5, ndcg10 = 0.0, 0.0

        try:
            for score, label in zip(activation, labels):
                auc += pl.metrics.functional.auroc(score, label)
                score = score.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                mrr += mrr_score(label, score)
                ndcg5 += ndcg_score(label, score, 5)
                ndcg10 += ndcg_score(label, score, 10)

            auroc = (auc / activation.shape[0]).item()
            mrr = (mrr / activation.shape[0]).item()
            ndcg5 = (ndcg5 / activation.shape[0]).item()
            ndcg10 = (ndcg10 / activation.shape[0]).item()
        except:
            auroc = 1.0
            mrr = 1.0
            ndcg5, ndcg10 = 1.0, 1.0

        return {'auroc': auroc, 'mrr': mrr, 'ndcg5': ndcg5, 'ndcg10': ndcg10}

    def validation_epoch_end(self, outputs):
        """
        inherit from pytorch lightning module, called after validation end
        :param outputs: List of outputs you defined in validation_step
        :return: metrics
        """
        mrr = torch.Tensor([x['mrr'] for x in outputs])
        auroc = torch.Tensor([x['auroc'] for x in outputs])
        ndcg5 = torch.Tensor([x['ndcg5'] for x in outputs])
        ndcg10 = torch.Tensor([x['ndcg10'] for x in outputs])

        logs = {
            "auroc": auroc.mean(),
            "mrr": mrr.mean(),
            "ndcg5": ndcg5.mean(),
            "ndcg10": ndcg10.mean(),
        }

        self.model.train()

        self.log_dict(logs, prog_bar=True)
