from abc import ABC
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=0.20, gamma=1.5, use_alpha=True, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
            # self.alpha = torch.tensor(alpha)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):

        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        # target_ = torch.zeros(target.size(0),self.class_num)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class BertForSequenceClassification(BertPreTrainedModel, ABC):
    def __init__(self, config, num_other, mid_dim=200):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_other = num_other
        self.config = config

        self.bert = BertModel(config, False)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)
        self.linear1 = nn.Sequential(
            nn.Linear(self.num_other, mid_dim),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(config.hidden_size, mid_dim),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(config.hidden_size, mid_dim),
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Linear(mid_dim, mid_dim, bias=False),
            # nn.LayerNorm(mid_dim, eps=1e-5)
        )

        # self.attn = nn.MultiheadAttention(mid_dim, 8, dropout=0.2, batch_first=True)
        self.visit_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(mid_dim, nhead=4, batch_first=True, norm_first=True),
            num_layers=4
        )

        self.classifier = nn.Linear(mid_dim, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        drug_input_ids: Optional[torch.Tensor] = None,
        drug_attention_mask: Optional[torch.Tensor] = None,
        other: Optional[torch.Tensor] = None,
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

        drug_outputs = self.bert(
            drug_input_ids,
            attention_mask=drug_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        drug_sequence_output = drug_outputs[0][:, 0]

        # pooled_output = outputs[1]
        # pooled_output = pooled_output.view((batch_size, visit_num, -1))[:, 0]
        # other_feature = self.linear(other)
        #
        # # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(torch.concat(
        #     (pooled_output, other_feature),
        #     dim=-1
        # ))

        sequence_output = outputs[0]
        sequence_output = self.linear2(sequence_output[:, 0].view((batch_size, visit_num, -1)))

        drug_sequence_output = self.linear3(drug_sequence_output[:, None, :])

        if other is not None:
            other_feature = self.linear1(other)

            output_feature = torch.concat(
                (sequence_output, drug_sequence_output, other_feature[:, None, :]),
                dim=1
            )
        else:
            output_feature = sequence_output

        output_feature = self.dense(output_feature)

        # output_feature, _ = self.attn(output_feature, output_feature, output_feature)
        if not self.training:
            output_feature = output_feature.half()
        output_feature = self.visit_encoder(output_feature)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(output_feature[:, 0])
        # logits = self.classifier(output_feature.max(1)[0])

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
