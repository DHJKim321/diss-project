import torch
import torch.nn as nn
from transformers import BertModel

class Bert(nn.Module):
    def __init__(self, bert_model, num_classes=2, use_dropout=True, dropout=0.3):
        super(Bert, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def to(self, device):
        self.bert = self.bert.to(device)
        self.classifier = self.classifier.to(device)
        return super().to(device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, pretrained_model_name="bert-base-uncased", num_classes=2, use_dropout=True, dropout=0.3):
        bert_model = BertModel.from_pretrained(pretrained_model_name)
        model = Bert(bert_model, num_classes=num_classes, use_dropout=use_dropout, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.eval()
        return model
