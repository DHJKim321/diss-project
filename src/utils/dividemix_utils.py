import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.mixture import GaussianMixture
from src.modules.losses import NegEntropy

def warmup_train(epoch_no, model, optimizer, dataloader):
    """
    Warmup training function for DivideMix
    """
    model.train()
    for _ in tqdm(enumerate(dataloader), desc="Warmup Training"):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = model(inputs)               
        loss = CrossEntropyLoss(outputs, labels)
        penalty = NegEntropy(outputs) # Details in the class documentation
        L = loss + penalty   
        L.backward()
        optimizer.step()
        print(f"Epoch {epoch_no}, Loss: {loss.item():.4f}, Penalty: {penalty.item():.4f}")


def eval_train(model, all_loss, eval_loader):    
    model.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    losses = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, _ in tqdm(enumerate(eval_loader), desc="Evaluating"):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CrossEntropyLoss(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
            print(f"Batch {batch_idx+1}/{num_iter}, Loss: {loss.mean().item()}")
                                    
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob, all_loss

def test(epoch_no, model1, model2, test_loader):
    preds, labels = [], []
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for _ in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)           
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f"Epoch {epoch_no}, Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
