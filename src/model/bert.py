import torch.nn as nn
import torch

class Bert(nn.Module):
    def __init__(self, bert_model, num_classes=2, use_dropout=True):
        super(Bert, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters(): # Ensure BERT parameters are frozen by default
            param.requires_grad = False
        self.dropout = nn.Dropout(0.3) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def freeze(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

    def train(self):
        self.bert.train()
        self.classifier.train()

    def eval(self):
        self.bert.eval()
        self.classifier.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.eval()

    def to_device(self, device):
        self.bert.to(device)
        self.classifier.to(device)
        return self