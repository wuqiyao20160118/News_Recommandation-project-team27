import torch
from torch.utils import data
import pytorch_lightning as pl
import bcolz
from pytorch_ranger import Ranger

from models.network import NRMS
from dataset.NewsDataset import NewsDataset
from utils.utils import ndcg_score, mrr_score
from glove import generate_glove_vocab


class NRMSModel(pl.LightningModule):
    def __init__(self, params):
        super(NRMSModel, self).__init__()
        self.hyperParams = params
        self.embedding_model = self.load_embedding()
        self.model = NRMS(params["model"], self.embedding_model)

    def load_embedding(self):
        """
        Load pre-trained glove embedding model
        :return: pre-trained embedding model
        """
        embedding_size = self.hyperParams["embedding_size"]
        max_vocab_size = self.hyperParams["max_vocab_size"]
        glove_path = self.hyperParams["max_vocab_size"]
        generate_glove_vocab(glove_path, embedding_size, max_vocab_size)
        embeddings = torch.Tensor(bcolz.open(f'{glove_path}/6B.'+str(embedding_size)+'.dat')[:])
        return embeddings

    def setup(self, stage: str):
        """
        inherit from pytorch lightning module
        :param stage: train / test
        :return:
        """
        if stage == "train" or stage == "test":
            news_dataset = NewsDataset(self.hyperParams, self.hyperParams["train_data_path"])
            data_size = len(news_dataset)
            train_size, val_size = int(data_size * 0.75), int(data_size * 0.15)
            test_size = data_size - train_size - val_size
            self.train_data, self.val_data, self.test_data = data.random_split(news_dataset, [train_size, val_size, test_size])
        else:
            print("Stage unknown!")
            return

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
                                     batch_size=self.hyperParams["batch_size"], shuffle=self.hyperParams["shuffle"])
        return val_loader

    def test_dataloader(self):
        """
        inherit from pytorch lightning module, assembling test dataloader
        :return:
        """
        test_loader = data.DataLoader(self.test_data, num_workers=self.hyperParams["num_workers"],
                                      batch_size=self.hyperParams["batch_size"], shuffle=self.hyperParams["shuffle"])
        return test_loader

    def configure_optimizers(
            self,
    ):
        """
        inherit from pytorch lightning module, configuring the optimizer
        :return: optimizer
        """
        return Ranger(self.parameters(), lr=self.hyperParams["lr"], weight_decay=1e-5)

    def forward(self):
        return None

    def training_step(self, batch, **kwargs):
        """
        inherit from pytorch lightning module, implements a mini-batch training step
        :param batch: batch data
        :param kwargs: Potentially used arguments
        :return: dictionary containing loss (prediction can also be contained)
        """
        clicks_title, _, candidates_title, _, labels = batch
        loss, _ = self.model(clicks_title, candidates_title, labels)
        return {"loss": loss}

    def training_epoch_end(self, output, **kwargs):
        """
        inherit from pytorch lightning module, implements after each epoch ends
        :param output: List of outputs you defined in training_step
        :param kwargs: Potentially used arguments
        :return: dictionary containing statistics
        """
        mean_loss = torch.stack([out['loss'] for out in output]).mean()
        self.model.eval()
        return {'log': {'loss': mean_loss}}

    def validation_step(self, batch, **kwargs):
        """
        inherit from pytorch lightning module, implements a mini-batch validation step
        :param batch: batch data
        :param kwargs: Potentially used arguments
        :return: dictionary containing evaluation metrics on training step
        """
        clicks_title, _, candidates_title, _, labels = batch
        with torch.no_grad():
            activation = self.model(clicks_title, candidates_title)
        mrr = 0.0
        auc = 0.0
        ndcg5, ndcg10 = 0.0, 0.0

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

        return {"log": logs}
