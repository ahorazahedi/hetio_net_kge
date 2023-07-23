import torch
from torch import nn

from torch.nn import functional as F


from collections import defaultdict
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch import cat, eq, int64, long, randperm, tensor, Tensor, zeros_like
from torch.utils.data import Dataset, DataLoader
import csv
import os

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import lightning as L


def transE(head, relation, tail, sample_type, margin):
    dist = head + relation - tail
    score = margin - torch.linalg.norm(dist, ord=1, dim=2)
    return score


class KGEmbedding(L.LightningModule):
    def __init__(self, hidden_dim, margin,

                 entity_filepath, relation_filepath,

                 function_type="transE"
                 ):
        super(KGEmbedding, self).__init__()

        self.function_type = function_type

        self.load_mappings(entity_filepath, relation_filepath)

        self.hidden_dim = hidden_dim
        self.e_dim = hidden_dim
        self.r_dim = hidden_dim

        self.margin = margin
        self.epsilon = 2
        self.embedding_range = (self.margin + self.epsilon) / hidden_dim

        self.entity_emb = nn.Parameter(torch.zeros(
            self.n_ent, self.e_dim, requires_grad=True))
        nn.init.uniform_(self.entity_emb, a=-
                         self.embedding_range, b=self.embedding_range)

        self.relation_emb = nn.Parameter(torch.zeros(
            self.n_rel, self.r_dim, requires_grad=True))
        nn.init.uniform_(self.relation_emb, a=-
                         self.embedding_range, b=-self.embedding_range)

        self.validation_step_metrics = []

    def forward(self, sample, sample_type):
        head, relation, tail = self.get_embeddings(sample, sample_type)
        return transE(head, relation, tail, sample_type, self.margin)

    def get_embeddings(self, sample, sample_type):
        if sample_type == "positive":
            head = self.entity_emb[sample[:, 0]].unsqueeze(1)
            relation = self.relation_emb[sample[:, 1]].unsqueeze(1)
            tail = self.entity_emb[sample[:, 2]].unsqueeze(1)

        elif sample_type == "negative_head":
            positive_tuple, negative_head_ent = sample
            bs, n_neg_samples = negative_head_ent.shape
            head = self.entity_emb[negative_head_ent.view(
                -1)].reshape(bs, n_neg_samples, self.e_dim)
            relation = self.relation_emb[positive_tuple[:, 1]].unsqueeze(1)
            tail = self.entity_emb[positive_tuple[:, 2]].unsqueeze(1)

        elif sample_type == "negative_tail":
            positive_tuple, negative_tail_ent = sample
            bs, n_neg_samples = negative_tail_ent.shape
            head = self.entity_emb[positive_tuple[:, 0]].unsqueeze(1)
            relation = self.relation_emb[positive_tuple[:, 1]].unsqueeze(1)
            tail = self.entity_emb[negative_tail_ent.view(
                -1)].reshape(bs, n_neg_samples, self.e_dim)

        return head, relation, tail

    def training_step(self, batch, batch_idx):
        sample, sample_type, labels = batch
        scores = self(sample, sample_type)
        loss = self.calculate_loss(scores, labels)
        self.log('train_loss', loss)
        return loss

    def calculate_loss(self, positive_sample, c_heads, c_tails):
        positive_sample_dist = self(positive_sample, "positive")
        positive_score = F.logsigmoid(positive_sample_dist)
        positive_sample_loss = -positive_score.mean()

        c_head_dist = self((positive_sample, c_heads), "negative_head")
        c_head_score = F.logsigmoid(-c_head_dist)
        c_head_loss = -c_head_score.mean()

        c_tail_dist = self((positive_sample, c_tails), "negative_tail")
        c_tail_score = F.logsigmoid(-c_tail_dist)
        c_tail_loss = -c_tail_score.mean()

        loss = ((2 * positive_sample_loss) + c_head_loss + c_tail_loss) / 4
        return loss

    def training_step(self, batch, batch_idx):

        positive_sample, c_heads, c_tails, _ = batch
        loss = self.calculate_loss(positive_sample, c_heads, c_tails)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        positive_sample, c_heads, c_tails, filter_bias = batch
        head_bias, tail_bias = filter_bias

        c_head_dist = self((positive_sample, c_heads),
                           'negative_head') + head_bias
        c_tail_dist = self((positive_sample, c_tails),
                           'negative_tail') + tail_bias

        head_arg_order = torch.argsort(c_head_dist, dim=1, descending=True)
        tail_arg_order = torch.argsort(c_tail_dist, dim=1, descending=True)

        true_head = positive_sample[:, 0]
        true_tail = positive_sample[:, 2]

        for ind in range(len(true_head)):
            true_head_rank = (
                head_arg_order[ind, :] == true_head[ind]).nonzero()
            true_head_rank = true_head_rank.item() + 1

            true_tail_rank = (
                tail_arg_order[ind, :] == true_tail[ind]).nonzero()
            true_tail_rank = true_tail_rank.item() + 1

            self.validation_step_metrics.append({
                'MRR': 1.0 / true_head_rank,
                'MR': float(true_head_rank),
                'HITS@1': 1.0 if true_head_rank <= 1 else 0.0,
                'HITS@3': 1.0 if true_head_rank <= 3 else 0.0,
                'HITS@10': 1.0 if true_head_rank <= 10 else 0.0
            })

    def on_validation_epoch_end(self):

        final_metrics = defaultdict(float)

        for metrics in self.validation_step_metrics:
            for key, val in metrics.items():
                final_metrics[key] += val

        for key, val in final_metrics.items():
            final_metrics[key] = val / len(self.validation_step_metrics)

        self.validation_step_metrics.clear()  # free memory

        self.log_dict(final_metrics)

        return final_metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=5e-4)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=40, max_epochs=1000)
        return [optimizer], [scheduler,]

    def predict_tail(self, head_name, relation_name, top_k=5):
        """Predict the top_k tail entities based on the head entity and relation."""

        head, relation = self.name_to_index(head_name, relation_name)

        head = int(head)
        relation = int(relation)

        head_emb = self.entity_emb[head, :].unsqueeze(0)
        relation_emb = self.relation_emb[relation, :].unsqueeze(0)
        projected_head = head_emb + relation_emb

        # Compute score with all entities
        scores = projected_head @ self.entity_emb.t()

        # Remove the head entity from the scores
        scores[:, head] = -float('inf')

        # Get the indices of the top_k highest scoring entities
        top_scores, tail_predictions = torch.topk(scores, k=top_k, dim=1)

        # Convert tensor of indices to list
        tail_predictions = tail_predictions.cpu().numpy().tolist()[0]

        tail_string = [self.id2entity[item] for item in tail_predictions]

        return tail_predictions, tail_string

    @staticmethod
    def load_dict(file_path, required_index=0):
        dict_item = dict()
        with open(file_path) as myfile:
            for index, line in enumerate(myfile):
                value = line.strip().split("\t")[required_index]
                dict_item[value] = int(index)
        return dict_item

    def load_mappings(self, entity_filepath, relation_filepath):
        self.entity2id = self.load_dict(entity_filepath)
        self.relation2id = self.load_dict(relation_filepath)
        # Also create reverse mappings
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)

    def name_to_index(self, entity_name, relation_name):
        entity_index = self.entity2id[entity_name]
        relation_index = self.relation2id[relation_name]
        return entity_index, relation_index
