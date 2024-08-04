from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, PreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings


class HcetModel(nn.Module):

    def __init__(self):
        super(HcetModel, self).__init__()
        # self.hidden_size = int(config.hidden_size / 2)
        self.hidden_size = 200
        self.rnn_hidden_size = 265
        self.num_labels = 1
        self.num_layers = 8
        self.demo_embeddings = nn.Embedding(120, self.hidden_size, padding_idx=0)
        self.topic_embeddings = nn.Embedding(100, self.hidden_size, padding_idx=0)
        self.disease_embeddings = nn.Embedding(2200, self.hidden_size, padding_idx=0)
        self.drug_embeddings = nn.Embedding(4590, self.hidden_size, padding_idx=0)
        self.weights = nn.Parameter(torch.ones((1, 5)), requires_grad=True)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.rnn_hidden_size),
            nn.ReLU()
        )
        # self.embeddings = BertEmbeddings(config)
        self.rnn_cell = nn.GRU(
            input_size=self.rnn_hidden_size,
            hidden_size=self.rnn_hidden_size,
            batch_first=True
        )
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.rnn_hidden_size, 1)

        self._init_weights(self)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def apply_emb(self, input_embs, input_mask, input_shape):
        input_embs = torch.relu(input_embs).view(input_shape + (-1,))
        if input_mask is not None:
            input_embs = input_embs * input_mask[:, :, :, None]
        input_embs = input_embs.sum(2)
        return input_embs

    def forward(self, demo=None, demo_mask=None,
                disease=None, disease_mask=None,
                topic=None, topic_mask=None,
                drug=None, drug_mask=None,
                labels=None):
        batch_size, visit_num, l_demo = demo.shape
        batch_size, visit_num, l_dis = disease.shape
        batch_size, visit_num, l_topic = topic.shape
        batch_size, visit_num, l_drug = drug.shape
        demo = demo.view((-1, l_demo))
        disease = disease.view((-1, l_dis))
        topic = topic.view((-1, l_topic))

        weights = self.weights.softmax(-1).repeat((batch_size, 1))[:, :, None, None]

        demo_hidden = self.apply_emb(
            self.demo_embeddings(demo), demo_mask, (batch_size, visit_num, l_demo)
        ) * weights[:, 0]
        dis_hidden = self.apply_emb(
            self.disease_embeddings(disease), disease_mask, (batch_size, visit_num, l_dis)
        ) * weights[:, 1]
        top_hidden = self.apply_emb(
            self.topic_embeddings(topic), topic_mask, (batch_size, visit_num, l_topic)
        ) * weights[:, 2]
        drug_hidden = self.apply_emb(
            self.drug_embeddings(drug), drug_mask, (batch_size, visit_num, l_drug)
        ) * weights[:, 3]
        ehr_hidden = self.linear(demo_hidden + dis_hidden + top_hidden + drug_hidden)

        _, ehr_hidden = self.rnn_cell(ehr_hidden)
        logits = self.classifier(ehr_hidden.squeeze(0)).squeeze(-1)

        outputs = (logits,)

        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # loss, logits
