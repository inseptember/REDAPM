from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, PreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings


class LstmForSequenceClassification(nn.Module):

    def __init__(self):
        super(LstmForSequenceClassification, self).__init__()
        # self.hidden_size = int(config.hidden_size / 2)
        self.input_size = 50
        self.hidden_size = 128
        self.num_labels = 1
        self.num_layers = 8
        # self.lstm = nn.LSTM(
        #     input_size=self.hidden_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     dropout=0.1,
        #     batch_first=True,
        #     bidirectional=True
        # )
        self.lstm = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            # num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2*self.hidden_size, self.num_labels)

        # self.init_weights()


    def apply_lstm(self, item_codes, item_length):
        packed_items = pack_padded_sequence(item_codes, item_length.double().cpu(), batch_first=True, enforce_sorted=False)
        complete_output, (h_n, _) = self.lstm(packed_items)
        output, _ = pad_packed_sequence(complete_output, batch_first=True)
        output = output[range(len(output)), item_length.long() - 1, :]
        return output  # shape: [batch_size, 2*hidden_size]

    def forward(self,item_codes, item_length,
        labels: Optional[torch.Tensor] = None):

        lstm_output = self.apply_lstm(item_codes, item_length)
        logits = self.classifier(self.dropout(lstm_output))

        outputs = (logits,)

        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # loss, logits
