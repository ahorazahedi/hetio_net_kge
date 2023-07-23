from collections import defaultdict
import numpy as np
import torch
from torch import cat, eq, int64, long, randperm, tensor, Tensor, zeros_like
from torch.utils.data import Dataset

import torch.nn.functional as F
import lightning as L

from utils import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class KnowledgeGraph(Dataset):
    def __init__(self, df=None, masked_kg=None, ent2ix=None, rel2ix=None, data_type="train", num_negative_samples=20, all_triplets=set()):
        self.ent2ix = ent2ix
        self.rel2ix = rel2ix

        self.data_type = data_type

        self.num_negative_samples = num_negative_samples

        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1

        if df is not None:
            self.n_facts = len(df)

            self.head_idx = tensor(df['from'].map(self.ent2ix).values).long()
            self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
            self.tail_idx = tensor(df['to'].map(self.ent2ix).values).long()

            subset = df[['from', 'rel', 'to']]
            all_triplets = {tuple(x)
                            for x in subset.to_numpy()}  # Change to set
            self.all_triplets = all_triplets

        else:
            self.n_facts = masked_kg['from'].shape[0]
            self.head_idx = masked_kg['from']
            self.tail_idx = masked_kg['to']
            self.relations = masked_kg['rel']
            self.all_triplets = all_triplets

        self.true_head_relation, self.true_relation_tail = self._get_true_head_tail_lists()

    def __len__(self):
        return self.n_facts

    def __getitem__(self, idx):
        head, relation, tail = self.head_idx[idx], self.relations[idx], self.tail_idx[idx]
        positive_sample = torch.LongTensor((head, relation, tail))

        if self.data_type == "train":
            true_heads = self.true_relation_tail[(
                relation.item(), tail.item())]
            true_tails = self.true_head_relation[(
                head.item(), relation.item())]

            # Change the way corrupted_heads and corrupted_tails are generated
            corrupted_heads = set()
            while len(corrupted_heads) < self.num_negative_samples:
                corrupted = np.random.randint(
                    self.n_ent, size=self.num_negative_samples)
                for c in corrupted:
                    if c not in true_heads:
                        corrupted_heads.add(c)

            corrupted_heads = torch.LongTensor(list(corrupted_heads))

            corrupted_tails = set()
            while len(corrupted_tails) < self.num_negative_samples:
                corrupted = np.random.randint(
                    self.n_ent, size=self.num_negative_samples)
                for c in corrupted:
                    if c not in true_tails:
                        corrupted_tails.add(c)

            corrupted_tails = torch.LongTensor(list(corrupted_tails))

            filter_bias = torch.LongTensor([0] * len(positive_sample))

        else:
            # Use set() instead of list() for all_triplets
            corrupted_heads = [(0, test_head) if (test_head, relation, tail) not in self.all_triplets
                               else (-1, head) for test_head in range(self.n_ent)]

            corrupted_tails = [(0, test_tail) if (head, relation, test_tail) not in self.all_triplets
                               else (-1, tail) for test_tail in range(self.n_ent)]
            corrupted_tails = [(0, test_tail)
                               for test_tail in range(self.n_ent)]

            corrupted_heads[head] = (0, head)
            corrupted_heads = torch.LongTensor(corrupted_heads)

            corrupted_tails = [(0, test_tail)
                               for test_tail in range(self.n_ent)]

            corrupted_tails[tail] = (0, tail)
            corrupted_tails = torch.LongTensor(corrupted_tails)

            filter_bias = (corrupted_heads[:, 0], corrupted_tails[:, 0])
            corrupted_heads = corrupted_heads[:, 1]
            corrupted_tails = corrupted_tails[:, 1]

        return positive_sample, corrupted_heads, corrupted_tails, filter_bias

    def _get_true_head_tail_lists(self):
        true_head_relation = defaultdict(set)
        true_relation_tail = defaultdict(set)
        for i in range(self.n_facts):

            #             Item Added To make It Iterable In Get Item Section
            head, relation, tail = self.head_idx[i],  self.relations[i],  self.tail_idx[i]
            true_head_relation[(head.item(), relation.item())].add(tail)
            true_relation_tail[(relation.item(), tail.item())].add(head.item())
        return true_head_relation, true_relation_tail

    def split_train_validation_test(self, share, validation=True):

        if validation is not False:
            mask_train, mask_validation, mask_test = self.get_mask(
                share, validation=True)
        else:
            mask_train, mask_test = self.get_mask(share, validation=True)

        train_kg = KnowledgeGraph(masked_kg={
            'from': self.head_idx[mask_train],
            'rel': self.relations[mask_train],
            'to': self.tail_idx[mask_train]
        }, ent2ix=self.ent2ix, rel2ix=self.rel2ix, data_type="train", all_triplets=self.all_triplets)
        test_kg = KnowledgeGraph(masked_kg={
            'from': self.head_idx[mask_test],
            'rel': self.relations[mask_test],
            'to': self.tail_idx[mask_test]
        }, ent2ix=self.ent2ix, rel2ix=self.rel2ix, data_type="validation", all_triplets=self.all_triplets)

        if validation is not False:
            validation_kg = KnowledgeGraph(masked_kg={
                'from': self.head_idx[mask_validation],
                'rel': self.relations[mask_validation],
                'to': self.tail_idx[mask_validation]
            }, ent2ix=self.ent2ix, rel2ix=self.rel2ix, data_type="test", all_triplets=self.all_triplets)
            return train_kg, validation_kg, test_kg

        return train_kg, test_kg

    def get_mask(self, share, validation=False):

        uniques_r, counts_r = self.relations.unique(return_counts=True)
        uniques_e, _ = cat((self.head_idx,
                            self.tail_idx)).unique(return_counts=True)
        mask = zeros_like(self.relations).bool()

        if validation:
            mask_val = zeros_like(self.relations).bool()

        for i, r in enumerate(uniques_r):
            rand = randperm(counts_r[i].item())
            sub_mask = eq(self.relations, r).nonzero(as_tuple=False)[:, 0]
            assert len(sub_mask) == counts_r[i].item()
            if validation:
                train_size, val_size, test_size = self.get_sizes(counts_r[i].item(),
                                                                 share=share,
                                                                 validation=True)

                mask[sub_mask[rand[:train_size]]] = True
                mask_val[sub_mask[rand[train_size:train_size + val_size]]] = True

            else:
                train_size, test_size = self.get_sizes(counts_r[i].item(),
                                                       share=share,
                                                       validation=False)
                mask[sub_mask[rand[:train_size]]] = True

        u = cat((self.head_idx[mask], self.tail_idx[mask])).unique()

        if len(u) < self.n_ent:
            missing_entities = tensor(list(set(uniques_e.tolist()) -
                                           set(u.tolist())), dtype=long)
            for e in missing_entities:
                sub_mask = ((self.head_idx == e) |
                            (self.tail_idx == e)).nonzero(as_tuple=False)[:, 0]
                rand = randperm(len(sub_mask))
                sizes = self.get_sizes(mask.shape[0],
                                       share=share,
                                       validation=validation)
                mask[sub_mask[rand[:sizes[0]]]] = True
                if validation:
                    mask_val[sub_mask[rand[:sizes[0]]]] = False

        if validation:
            assert not (mask & mask_val).any().item()
            return mask, mask_val, ~(mask | mask_val)
        else:
            return mask, ~mask

    @staticmethod
    def get_sizes(count, share, validation=False):

        if count == 1:
            if validation:
                return 1, 0, 0
            else:
                return 1, 0
        if count == 2:
            if validation:
                return 1, 1, 0
            else:
                return 1, 1

        n_train = int(count * share)
        assert n_train < count
        if n_train == 0:
            n_train += 1

        if not validation:
            return n_train, count - n_train
        else:
            if count - n_train == 1:
                n_train -= 1
                return n_train, 1, 1
            else:
                n_val = int(int(count - n_train) / 2)
                return n_train, n_val, count - n_train - n_val


def custom_collate_fn(batch):
    max_heads = max([len(sample[1]) for sample in batch])
    max_tails = max([len(sample[2]) for sample in batch])

    batch_heads = [F.pad(sample[1], (0, max_heads - len(sample[1])))
                   for sample in batch]
    batch_tails = [F.pad(sample[2], (0, max_tails - len(sample[2])))
                   for sample in batch]

    positive_samples = torch.stack([sample[0] for sample in batch], 0)
    corrupted_heads = torch.stack(batch_heads, 0)
    corrupted_tails = torch.stack(batch_tails, 0)
    filter_biases = (torch.stack([sample[3][0] for sample in batch], 0),
                     torch.stack([sample[3][1] for sample in batch], 0))

    return positive_samples, corrupted_heads, corrupted_tails, filter_biases


class KGDataModule(L.LightningDataModule):

    def __init__(self, df_path="./hetionet/het.tsv", entity2id_path='hetionet/entity2id.tsv', relation2id='hetionet/relation2id.tsv', batch_size=64, validation_batch_size=32, num_workers=6):
        super().__init__()

        self.df = pd.read_csv(df_path, sep="\t")
        self.entity2id = load_dict(entity2id_path)
        self.relation2id = load_dict(relation2id)

        num_entities = len(self.entity2id)
        num_relations = len(self.relation2id)

        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size

        self.num_workers = num_workers

    def setup(self, stage: str):
        kg = KnowledgeGraph(df=self.df, ent2ix=self.entity2id,
                            rel2ix=self.relation2id)

        self.train_kg, self.validation_kg, self.test_kg = kg.split_train_validation_test(
            0.8, True)
        print("item of Datasets", len(self.train_kg),
              len(self.validation_kg), len(self.test_kg))

    def train_dataloader(self):
        return DataLoader(
            self.train_kg, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, collate_fn=custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(self.validation_kg, batch_size=self.validation_batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_kg, batch_size=self.validation_batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)
