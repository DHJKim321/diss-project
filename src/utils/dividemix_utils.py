import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.mixture import GaussianMixture
import numpy as np
import random
torch.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True

def warmup_train(epoch_no, model, optimizer, warmup_loader, criterion, negentropy, device='cuda'):
    """
    Warmup training function for DivideMix
    """
    model.train()
    for _, batch in tqdm(enumerate(warmup_loader), desc="Warmup Training", total=len(warmup_loader)):      
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        penalty = negentropy(outputs) # Details in the class documentation
        L = loss + penalty
        L.backward()
        optimizer.step()
        # tqdm.write(f"Epoch {epoch_no}, Loss: {loss.item():.4f}, Penalty: {penalty.item():.4f}")

def train(epoch_no, model1, model2, optimizer, semiloss, labelled_loader, unlabelled_loader, warmup_epochs, batch_size=64, temperature=0.5, alpha=0.5, num_class=2, device='cuda'):
    """
    Training function for DivideMix.
    This function implements the MixMatch algorithm with label co-guessing and co-refinement.
    It uses two models (model1 and model2) to refine the labels of labelled samples and
    guess the labels of unlabelled samples.

    Args:
        epoch_no: Current epoch number.
        model1: Model to be trained
        model2: Model to be used for Label Co-guessing
        optimizer: Optimizer for model1
        labelled_loader: DataLoader for labelled samples
        unlabelled_loader: DataLoader for unlabelled samples
        batch_size: Batch size for training
        temperature: Temperature for label sharpening
        alpha: Alpha parameter for MixMatch
        num_class: Number of classes in the dataset
        device: Device to run the training on (default is 'cuda')
    """
    model1.train()
    model2.eval()
    
    unlabelled_train_iter = iter(unlabelled_loader)
    num_iter = (len(labelled_loader.dataset)//batch_size)+1
    # MixMatch requires the same number of labelled and unlabelled samples in each batch.
    for batch_idx, batch in tqdm(enumerate(labelled_loader), desc="Training", total=num_iter):
        input_ids_x1 = batch['input_ids_1'].to(device)
        input_ids_x2 = batch['input_ids_2'].to(device)
        attention_mask_x1 = batch['attention_mask_1'].to(device)
        attention_mask_x2 = batch['attention_mask_2'].to(device)
        labels_x = batch['labels'].to(device)
        prob = batch['probability'].to(device)
        try:
            batch_u = next(unlabelled_train_iter)
        except:
            unlabelled_train_iter = iter(unlabelled_loader)
            batch_u = next(unlabelled_train_iter)
        input_ids_u1 = batch_u['input_ids_1'].to(device)
        attention_mask_u1 = batch_u['attention_mask_1'].to(device)
        input_ids_u2 = batch_u['input_ids_2'].to(device)
        attention_mask_u2 = batch_u['attention_mask_2'].to(device)
        batch_size = input_ids_x1.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, 2, device=device).scatter_(1, labels_x.view(-1, 1), 1)
        prob = prob.view(-1,1).float().to(device)

        with torch.no_grad():
            # ---- Label Co-guessing (Unlabelled Samples) ----
            outputs_u11 = model1(input_ids_u1, attention_mask_u1)
            outputs_u12 = model1(input_ids_u2, attention_mask_u2)
            outputs_u21 = model2(input_ids_u1, attention_mask_u1)
            outputs_u22 = model2(input_ids_u2, attention_mask_u2)
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/temperature) # Temparature Sharpening
            
            labels_u = ptu / ptu.sum(dim=1, keepdim=True) # Normalize
            labels_u = labels_u.detach()       
            
            # ----- Label Co-refinement (Labelled Samples) ----
            outputs_x = model1(input_ids_x1, attention_mask_x1)
            outputs_x2 = model1(input_ids_x2, attention_mask_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2 # Average the outputs of the two models
            px = prob * labels_x + (1 - prob) * px # prob tells us the likelihood of the label being correct using the GMM's cluster probability
            # labels_x is the ground-truth, px is the average of the two models' predictions
            ptx = px**(1/temperature) # Temparature Sharpening
                       
            labels_x = ptx / ptx.sum(dim=1, keepdim=True) # Normalize
            labels_x = labels_x.detach()
        
        # ---- MixMatch ----
        l = np.random.beta(alpha, alpha)        
        l = max(l, 1-l)
        
        # Calculate embeddings for MixMatch
        layer_index = np.random.choice([7, 9, 12]) # Based on the paper below, these layers contain the richest syntactic and semantic information
        layer_index -= 1 # Convert to zero-based index

        # Get embeddings for labelled samples
        # https://arxiv.org/pdf/2004.12239 - MixText/TMix
        # Instead of using final embeddngs, we use the model's hidden state representations
        embedding_x1 = model1.get_embedding_at_layer(input_ids_x1, attention_mask_x1, layer_index=layer_index)
        embedding_x2 = model1.get_embedding_at_layer(input_ids_x2, attention_mask_x2, layer_index=layer_index)

        # Get embeddings for unlabelled samples
        embedding_u1 = model1.get_embedding_at_layer(input_ids_u1, attention_mask_u1, layer_index=layer_index)
        embedding_u2 = model1.get_embedding_at_layer(input_ids_u2, attention_mask_u2, layer_index=layer_index)

        # Concatenate embeddings and labels for MixMatch
        # all_inputs.shape = [batch_size * 4, seq_len, hidden_size]
        all_inputs = torch.cat([embedding_x1, embedding_x2, embedding_u1, embedding_u2], dim=0) # Concatenate embeddings
        all_labels = torch.cat([labels_x, labels_x, labels_u, labels_u], dim=0) # Soft labels from refinement/guessing

        idx = torch.randperm(all_inputs.size(0)) # Generates random permutation of indices

        # This forms random MixUp pairs
        input_a, input_b = all_inputs, all_inputs[idx]
        label_a, label_b = all_labels, all_labels[idx]

        # Interpolate inputs and labels
        # Only use half of the inputs to avoid excessive memory usage
        # mixed_input.shape = [batch_size * 2, seq_len, hidden_size]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_labels = l * label_a + (1 - l) * label_b
        final_attention_mask = torch.ones_like(mixed_input[:, :, 0]).to(device) # Apply all-one attention mask following the paper
        # mixed_input.shape = (batch_size*2, seq_length, hidden_size)
        logits = model1.forward_from_layer(mixed_input, final_attention_mask, layer_index=layer_index+1)

        # Split into labelled and unlabelled
        logits_x = logits[:batch_size]
        targets_x = mixed_labels[:batch_size]
        logits_u = logits[batch_size:]
        targets_u = mixed_labels[batch_size:]
        
        # Calculate individual losses
        Lx, Lu, lambda_u_val = semiloss(logits_x, targets_x, logits_u, targets_u, epoch_no, warmup_epochs)
        
        # Calculate regularization penalty - (Tanaka et al. 2018), (Arazo et al. 2019)
        prior = torch.full((num_class,), 1 / num_class, device=device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        # Combine losses
        loss = Lx + lambda_u_val * Lu + penalty

        # ---- Optimizer Step ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Debugging
        # for name, p in model1.named_parameters():
        #     if p.grad is None:
        #         tqdm.write(f"Parameter {name} has no gradient.")

        # if batch_idx == 0:            # first batch of epoch
        #     print("optim step =", optimizer.state_dict()['state'][next(iter(optimizer.state_dict()['state']))]['step'])
        # ----

        # tqdm.write(f"Epoch {epoch_no}, Batch {batch_idx+1}/{num_iter}, Lx {Lx.item():.4f}, Lu {Lu.item():.4f}, Pentalty {penalty.item():.4f}, loss {loss.item():.4f}")

def eval_train(model, all_loss, criterion, eval_loader, device='cuda'):    
    model.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    losses = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(eval_loader), desc="Evaluating Training Data", total=num_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            index = batch['index']
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu()
            for b in range(input_ids.size(0)):
                losses[index[b]]=loss[b] # losses.shape = [batch_size,]
            # tqdm.write(f"Batch {batch_idx+1}/{num_iter}, Loss: {loss.mean().item()}")

    raw_losses = losses.detatch().cpu().numpy()
    losses = (losses-losses.min())/(losses.max()-losses.min()) # Normalize losses to [0, 1]
    all_loss.append(losses)

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1,1).cpu().numpy() # input_loss.shape = [n_samples, 1]
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) # prob.shape = [n_samples, n_components]
    prob = prob[:,gmm.means_.argmin()] # prob.shape = [n_samples], gmm.means_.shape = [n_components, 1] - the centre of each component cluster
    # I guess ^ is just a way to not use a hard-coded index? You can still calculate probabilities for both indices since n_components=2
    # gmm.means_.argmin() returns the index of the component with the smallest mean
    return prob, all_loss, raw_losses

def test(model1, model2, test_loader, device='cuda'):
    preds, all_labels = [], []
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs1 = model1(input_ids, attention_mask)
            outputs2 = model2(input_ids, attention_mask)
            outputs = outputs1 + outputs2 # outputs.shape = [batch_size, num_classes]
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='weighted')
    return acc, f1, preds, all_labels
