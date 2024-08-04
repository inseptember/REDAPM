from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, PreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings


class LstmForSequenceClassification(PreTrainedModel):

    def __init__(self, config):
        super(LstmForSequenceClassification, self).__init__(config)
        # self.hidden_size = int(config.hidden_size / 2)
        config.hidden_size = 200
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.num_layers = 8
        self.max_pooling = config.max_pooling
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size, padding_idx=config.pad_token_id)
        # self.embeddings = BertEmbeddings(config)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            # num_layers=self.num_layers,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2*self.hidden_size, config.num_labels)

        self.init_weights()

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


    def apply_lstm(self, input_ids):
        # input id input_ids
        # output is the complete output of lstm
        # for efficiency: improve this function (sorting? packing?)
        batch_size, max_seq_len = input_ids.shape
        sent_len = torch.sum(input_ids!=0, dim=1)
        sorted_sent_len, forward_sort_order = torch.sort(sent_len, descending=True)
        _, backward_sort_order = torch.sort(forward_sort_order)
        sorted_batch = self.embeddings(input_ids)[forward_sort_order]
        packed_batch = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_batch, sorted_sent_len.cpu(), batch_first=True
        )
        complete_output, (h_n, _) = self.lstm(packed_batch)
        if self.max_pooling:
            output = torch.nn.utils.rnn.pad_packed_sequence(complete_output)[0]  # basically unpack it
            output = output.max(dim=0)[0][backward_sort_order]
            # max: max pooling along the seq_len dimension
            # [backward_order]: recover the order before sorting
        else:
            output = h_n.permute(1,0,2)[backward_sort_order].view(batch_size, self.num_layers, 2, -1)[:,-1,:,:].view(batch_size, -1)
            # permute: change batch to the first index
            # [backward_order]: recover the order before sorting
            # view: separate num_layers with num_directions
            # [...]: take only the last layer
            # view: merge both directions
        return output  # shape: [batch_size, 2*hidden_size]

    def forward(self,input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        drug_input_ids: Optional[torch.Tensor] = None,
        drug_attention_mask: Optional[torch.Tensor] = None,
        other: Optional[torch.Tensor] = None,
        visit_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):

        lstm_output = self.apply_lstm(input_ids)
        logits = self.classifier(self.dropout(lstm_output))

        outputs = (logits,)
        batch_size = logits.shape[0]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        outputs = (loss,) + outputs

        return outputs  # loss, logits
