import torch
import torch.nn as nn
from transformers import BertModel

class Bert(nn.Module):
    def __init__(self, bert_model, head_type='linear', num_classes=2, use_dropout=True, dropout=0.3, use_hidden_state=False):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout) if use_dropout else nn.Identity()
        self.use_hidden_state = use_hidden_state
        if head_type == "linear":
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        elif head_type == 'lstm':
            self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size // 2, 
                               num_layers=1, bidirectional=True, batch_first=True)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        elif head_type == 'cnn':
            self.conv1 = nn.Conv1d(self.bert.config.hidden_size, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.classifier = nn.Linear(64 * (self.bert.config.max_position_embeddings // 2), num_classes)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if hasattr(self, 'lstm'):
            lstm_out, _ = self.lstm(last_hidden_state)
            pooled_output = lstm_out[:, 0, :]
        elif hasattr(self, 'conv1'):
            x = last_hidden_state.permute(0, 2, 1)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            pooled_output = x
        else:
            pooled_output = outputs.pooler_output

        return self.dropout(pooled_output)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        if hasattr(self, 'lstm'):
            lstm_out, _ = self.lstm(last_hidden_state)
            pooled_output = lstm_out[:, 0, :]
        elif hasattr(self, 'conv1'):
            x = last_hidden_state.permute(0, 2, 1)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            pooled_output = x
        else:
            pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def forward_from_layer(self, mixed_hidden_state, attention_mask, layer_index):
        # Manually run the remaining layers of the encoder
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, attention_mask.shape, attention_mask.device)
        # Go from mix_layer to end
        for layer_module in self.bert.encoder.layer[layer_index:]:
            mixed_hidden_state = layer_module(mixed_hidden_state, attention_mask=extended_attention_mask, head_mask=None)[0]
        pooled_output = mixed_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def get_embedding_at_layer(self, input_ids, attention_mask, layer_index):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        return hidden_states[layer_index] # hidden_state.shape = [batch_size, seq_length, hidden_size]

    def forward_mixup(self, input_ids_x1, attention_mask_x1, input_ids_x2, attention_mask_x2, labels_x, 
                      input_ids_u1, attention_mask_u1, input_ids_u2, attention_mask_u2, labels_u, layer_index,
                      mix_lambda, device='cuda'):
        
        # Get initial embeddings from BERT
        emb_x1 = self.bert.embeddings(input_ids_x1)
        emb_x2 = self.bert.embeddings(input_ids_x2)
        emb_u1 = self.bert.embeddings(input_ids_u1)
        emb_u2 = self.bert.embeddings(input_ids_u2)

        hidden_x1 = emb_x1
        hidden_x2 = emb_x2
        hidden_u1 = emb_u1
        hidden_u2 = emb_u2

        # Get extended attention masks
        extended_attention_mask_x1 = self.bert.get_extended_attention_mask(attention_mask_x1, attention_mask_x1.shape, device)
        extended_attention_mask_x2 = self.bert.get_extended_attention_mask(attention_mask_x2, attention_mask_x2.shape, device)
        extended_attention_mask_u1 = self.bert.get_extended_attention_mask(attention_mask_u1, attention_mask_u1.shape, device)
        extended_attention_mask_u2 = self.bert.get_extended_attention_mask(attention_mask_u2, attention_mask_u2.shape, device)

        # Iterate through from layer 0 to layer_index (exclusive)
        encoder_layers = self.bert.encoder.layer
        for i in range(layer_index):
            layer_module = encoder_layers[i]
            hidden_x1 = layer_module(hidden_x1, attention_mask=extended_attention_mask_x1)[0]
            hidden_x2 = layer_module(hidden_x2, attention_mask=extended_attention_mask_x2)[0]
            hidden_u1 = layer_module(hidden_u1, attention_mask=extended_attention_mask_u1)[0]
            hidden_u2 = layer_module(hidden_u2, attention_mask=extended_attention_mask_u2)[0]

        # Concatenate all embeddings and labels
        all_inputs = torch.cat([hidden_x1, hidden_x2, hidden_u1, hidden_u2], dim=0) # Concatenate embeddings
        all_labels = torch.cat([labels_x, labels_x, labels_u, labels_u], dim=0) # Soft labels from refinement/guessing

        # Generates random permutation of indices
        idx = torch.randperm(all_inputs.size(0))

        # This forms random MixUp pairs
        input_a, input_b = all_inputs, all_inputs[idx]
        label_a, label_b = all_labels, all_labels[idx]

        # Interpolate inputs and labels
        # Only use half of the inputs to avoid excessive memory usage
        # mixed_input.shape = [batch_size * 2, seq_len, hidden_size]
        mixed_hidden = mix_lambda * input_a + (1 - mix_lambda) * input_b
        mixed_labels = mix_lambda * label_a + (1 - mix_lambda) * label_b
        attention_mask = torch.ones(mixed_hidden.size(0), mixed_hidden.size(1), dtype=torch.long, device=device)
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, attention_mask.shape, device)

        # Run the remaining layers from layer_index to the end
        for i in range(layer_index, len(self.bert.encoder.layer)):
            mixed_hidden = self.bert.encoder.layer[i](mixed_hidden, attention_mask=extended_attention_mask)[0]

        # Get the pooled output
        pooled_output = mixed_hidden[:, 0]
        pooled_output = self.dropout(pooled_output)

        # Classify the pooled output
        logits = self.classifier(pooled_output)
        return logits, mixed_labels

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

    def to_device(self, device):
        self.bert = self.bert.to(device)
        self.classifier = self.classifier.to(device)
        return super().to(device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, device, head_type, pretrained_model_name="bert-base-uncased", num_classes=2, use_dropout=True, dropout=0.3):
        model = Bert(pretrained_model_name, head_type, num_classes=num_classes, use_dropout=use_dropout, dropout=dropout)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
