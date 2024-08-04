from abc import ABC
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F

class LatentLayer(nn.Module):
    def __init__(self, mid_dim):
        super(LatentLayer, self).__init__()
        self.genDim = mid_dim
        latent_dim = mid_dim
        self.linear = nn.Linear(latent_dim, self.genDim)
        self.bn = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation = nn.ReLU()
    def forward(self, x):
        residual = x
        temp = self.activation(self.linear(x))
        out = self.bn(temp + residual)
        return out

class Latent(nn.Module):
    def __init__(self, mid_dim, layer_num=2):
        super(Latent, self).__init__()
        self.layers = nn.ModuleList([LatentLayer(mid_dim) for _ in range(layer_num)])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class BertForSequenceClassification(BertPreTrainedModel, ABC):
    def __init__(self, config, num_other, mid_dim=200, laten_layer=4):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_other = num_other
        self.config = config

        self.bert = BertModel(config, False)
        self.linear1 = nn.Sequential(
            nn.Linear(self.num_other, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.Dropout(0.1),
            Latent(mid_dim, laten_layer)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(config.hidden_size, mid_dim),
            nn.ReLU()
        )
        self.visit_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(mid_dim, nhead=8, batch_first=True, norm_first=True),
            num_layers=4
        )

        self.classifier = nn.Linear(mid_dim, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        other: Optional[torch.Tensor] = None,
        visit_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, visit_num, seq_length = input_ids.shape

        input_ids = input_ids.view((-1, seq_length))
        attention_mask = attention_mask.view((-1, seq_length))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.linear2(sequence_output[:, 0].view((batch_size, visit_num, -1)))

        # drug_sequence_output = self.linear3(drug_sequence_output[:, None, :])

        if other is not None:
            other_feature = self.linear1(other)

            output_feature = torch.concat(
                (sequence_output, other_feature[:, None, :]),
                dim=1
            )
            visit_mask = torch.concat((visit_mask, torch.ones((batch_size, 1)).to(visit_mask.device)), dim=-1)
        else:
            output_feature = sequence_output
        if not self.training:
            output_feature = output_feature.half()
        output_feature = self.visit_encoder(output_feature, src_key_padding_mask=(visit_mask == 0))

        logits = self.classifier(output_feature[:, 0])

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(torch.tensor([0.1, 1]).float().to(logits.device))
            loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        if not return_dict:
            output = (logits,) + output_feature[:, 0]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output_feature[:, 0],
            attentions=None,
        )
