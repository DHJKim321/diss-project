import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch.nn as nn
from transformers import BertModel

class MixTextBert(nn.Module):
    def __init__(self, bert_model, head_type='linear', num_classes=2, use_dropout=True, dropout=0.3):
        super(MixTextBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout) if use_dropout else nn.Identity()
        if head_type == "linear":
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        elif head_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        
    def run_until(self, input_ids, attention_mask, layer_index, device='cuda'):
        hidden = self.bert.embeddings(input_ids)
        extended_mask = self.bert.get_extended_attention_mask(attention_mask, attention_mask.shape, device)
        for i in range(layer_index+1): # Run until the output of the layer index
            hidden = self.bert.encoder.layer[i](hidden, extended_mask)[0]
        return hidden, extended_mask
    
    def run_from(self, hidden_states, extended_mask, layer_index):
        for i in range(layer_index+1, len(self.bert.encoder.layer)): # Start from the next layer
            hidden_states = self.bert.encoder.layer[i](hidden_states, extended_mask)[0]
        return hidden_states
        
    def forward_mix(self, input_a, attention_mask_a, input_b, attention_mask_b, l, layer_index):
        hidden_a, extended_mask_a = self.run_until(input_a, attention_mask_a, layer_index)
        hidden_b, _ = self.run_until(input_b, attention_mask_b, layer_index)
        hidden_mixed = l * hidden_a + (1 - l) * hidden_b

        hidden_output = self.run_from(hidden_mixed, extended_mask_a, layer_index)
        pooled_output = hidden_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits