import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
import numpy as np
from tqdm import tqdm

def train(labelled_trainloader, unlabelled_trainloader, model, optimizer, criterion, epoch, n_labels, T, alpha, device):
    labeled_train_iter = iter(labelled_trainloader)
    unlabeled_train_iter = iter(unlabelled_trainloader)
    model.train()

    total_loss = 0
    total_samples = 0

    for batch_idx in tqdm(range(len(labelled_trainloader))):
        # Get labelled data
        batch_x = next(labeled_train_iter)
        input_ids_x = batch_x['input_ids'].to(device)
        attention_mask_x = batch_x['attention_mask'].to(device)
        labels_x = batch_x['labels'].to(device)
        batch_size = input_ids_x.size(0)
        labels_x = torch.zeros(batch_size, n_labels, device=device).scatter_(1, labels_x.view(-1, 1), 1)

        # Get unlabelled data
        batch_u = next(unlabeled_train_iter)
        input_ids_u_orig = batch_u['input_ids_orig'].to(device)
        attention_mask_u_orig = batch_u['attention_mask_orig'].to(device)
        input_ids_u_aug_1 = batch_u['input_ids_aug_1'].to(device)
        attention_mask_u_aug_1 = batch_u['attention_mask_aug_1'].to(device)
        input_ids_u_aug_2 = batch_u['input_ids_aug_2'].to(device)
        attention_mask_u_aug_2 = batch_u['attention_mask_aug_2'].to(device)

        with torch.no_grad():
            # Predict labels for unlabeled data.
            outputs_u1 = model(input_ids_u_aug_1, attention_mask_u_aug_1)
            outputs_u2 = model(input_ids_u_aug_2, attention_mask_u_aug_2)
            outputs_ori = model(input_ids_u_orig, attention_mask_u_orig)

            p = 0.25 * torch.softmax(outputs_u1, dim=1) + 0.25 * torch.softmax(outputs_u2, dim=1) + 0.5 * torch.softmax(outputs_ori, dim=1)
            pt = p**(1/T)
            labels_u = pt / pt.sum(dim=1, keepdim=True)
            labels_u = labels_u.detach()

        l = np.random.beta(alpha, alpha)        
        l = max(l, 1-l)

        # Calculate embeddings for MixMatch
        layer_index = np.random.choice([7, 9, 12]) # Based on the paper below, these layers contain the richest syntactic and semantic information
        layer_index -= 1 # Convert to zero-based index

        all_inputs = torch.cat([input_ids_x, input_ids_u_aug_1, input_ids_u_aug_2, input_ids_u_orig], dim=0)
        all_masks = torch.cat([attention_mask_x, attention_mask_u_aug_1, attention_mask_u_aug_2, attention_mask_u_orig], dim=0)
        all_labels = torch.cat([labels_x, labels_u, labels_u, labels_u], dim=0)

        perm_idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[perm_idx]
        attention_mask_a, attention_mask_b = all_masks, all_masks[perm_idx]
        labels_a, labels_b = all_labels, all_labels[perm_idx]

        # Mix hidden state representations
        logits = model.forward_mix(input_a, attention_mask_a, input_b, attention_mask_b, l, layer_index)
        mixed_labels = l * labels_a + (1 - l) * labels_b

        # Calculate loss
        # Split the logits back into labelled, augmented unlabelled, and original unlabelled parts
        logits_x = logits[:batch_size]
        mixed_labels_x = mixed_labels[:batch_size]
        logits_u = logits[batch_size:]
        mixed_labels_u = mixed_labels[batch_size:]

        Lx, Lu, lambda_u1 = criterion(logits_x, mixed_labels_x, logits_u, mixed_labels_u, epoch+batch_idx/len(labelled_trainloader))
        loss = Lx + lambda_u1 * Lu
        total_loss += loss.item()
        total_samples += batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / total_samples

def validate(val_loader, model, criterion, device='cuda'):
    model.eval()
    loss_total = 0
    total_sample = 0
    acc_total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(inputs, attention_mask)
            loss = criterion(logits, labels)

            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total