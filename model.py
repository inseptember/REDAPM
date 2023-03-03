import copy
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Callable, Any

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, ModuleList
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


@dataclass
class DepClassifierOutput(SequenceClassifierOutput):
    visit_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    visit_attentions: Optional[Tuple[torch.FloatTensor]] = None
    drug_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    drug_attentions: Optional[Tuple[torch.FloatTensor]] = None
    device: Optional[str] = None



class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Any, Any]:
        x = src
        hidden, att = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + hidden
        x = x + self._ff_block(self.norm2(x))

        return x, att

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tuple[Any, Any]:
        x, att = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout1(x), att

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                return_attention=False) -> Tuple[
        Union[Tensor, Any], Union[Optional[Tuple[Any]], Any], Union[Optional[Tuple[Any]], Any]]:

        output = src
        hidden_layers = None
        attention_weights = None
        for mod in self.layers:
            output, att = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if hidden_layers is None:
                hidden_layers = (output,)
                attention_weights = (att,)
            else:
                hidden_layers = hidden_layers + (output, )
                attention_weights = attention_weights + (att, )
        return output, hidden_layers, attention_weights


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
        self.visit_encoder = TransformerEncoder(
            TransformerEncoderLayer(mid_dim, nhead=4, batch_first=True, norm_first=True),
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
            labels: Optional[torch.Tensor] = None,
            # token_type_ids: Optional[torch.Tensor] = None,
            # position_ids: Optional[torch.Tensor] = None,
            # head_mask: Optional[torch.Tensor] = None,
            # inputs_embeds: Optional[torch.Tensor] = None,
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
        input_ids = input_ids.int()
        attention_mask = attention_mask.int()
        drug_input_ids = drug_input_ids.int()
        drug_attention_mask = drug_attention_mask.int()
        # other = data_input['other']
        # labels = data_input['labels']
        return_dict = True
        output_attentions = True
        output_hidden_states = True
        head_mask = None



        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, visit_num, seq_length = input_ids.shape

        input_ids = input_ids.view((-1, seq_length))
        attention_mask = attention_mask.view((-1, seq_length))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
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
        # if not self.training:
        #     output_feature = output_feature.half()
        output_feature, visit_hidden_states, visit_attentions = self.visit_encoder(output_feature)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(output_feature[:, 0])
        # logits = self.classifier(output_feature.max(1)[0])

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(torch.tensor([0.1, 1]).float().to(logits.device))
            loss_fct = CrossEntropyLoss(reduction='none')
            # loss_fct = FocalLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        # if not return_dict:
        #     output = (logits,) + output_feature[:, 0]
        #     return ((loss,) + output) if loss is not None else output
        #
        # return DepClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     drug_hidden_states=drug_outputs.hidden_states,
        #     drug_attentions=drug_outputs.attentions,
        #     visit_hidden_states=visit_hidden_states,
        #     visit_attentions=visit_attentions,
        #     device=loss.device
        # )
        return logits.softmax(-1)
