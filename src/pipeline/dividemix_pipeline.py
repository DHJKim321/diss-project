import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from torch.optim import AdamW
import torch.nn as nn
from src.model.bert import Bert
from src.utils.data_utils import *
from src.utils.eval_utils import save_evaluation, add_predictions_to_data, evaluate_model, save_loss_as_df
from src.utils.dividemix_utils import warmup_train, train, eval_train, test
from src.data.dataloader.DivideMixDataloader import DivideMixDataloader
from src.modules.losses import SemiLoss, NegEntropy
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

import random
# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    # ------------ Load environment variables ------------
    load_dotenv()

    # ---- Data Paths ----
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    train_file = os.getenv("TRAIN_FILE")
    train_data_path = os.getenv("TRAIN_DATA_PATH")
    test_file = os.getenv("TEST_FILE")
    test_data_path = os.getenv("TEST_DATA_PATH")
    data_save_path = os.getenv("DATA_SAVE_PATH")
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    dataset = os.getenv("DATASET")
    if dataset == 'imdb':
        test_file = os.getenv("TRAIN_FILE")
    # Model Training Variables
    bert_model = os.getenv("BERT_MODEL")
    batch_size = int(os.getenv("BATCH_SIZE"))
    learning_rate = float(os.getenv("LEARNING_RATE"))
    warmup_epochs = int(os.getenv("WARMUP_EPOCHS"))
    epochs = int(os.getenv("EPOCHS"))
    head_type = os.getenv("HEAD_TYPE")
    # DivideMix Variables
    alpha = float(os.getenv("ALPHA"))
    penalty_val = float(os.getenv("PENALTY_VAL"))
    lambda_u = float(os.getenv("LAMBDA_U"))
    p_threshold = float(os.getenv("P_THRESHOLD"))
    temperature = float(os.getenv("SHARPENING_TEMPERATURE"))
    num_workers = int(os.getenv("NUM_WORKERS"))
    momentum = float(os.getenv("MOMENTUM"))
    weight_decay = float(os.getenv("WEIGHT_DECAY"))
    augmentation = os.getenv("AUGMENTATION")
    noise_ratio = float(os.getenv("NOISE_RATIO"))
    rampup = float(os.getenv("RAMPUP"))
    # Early-Late Dropout Variables
    dropout_type = os.getenv("DROPOUT_TYPE")
    p_early = float(os.getenv("P_EARLY"))
    p_late = float(os.getenv("P_LATE"))
    # GMM Early Stopping Variables
    gmm_patience = int(os.getenv("GMM_PATIENCE"))
    delta_sep_tol = float(os.getenv("DELTA_SEP_TOL")) # Minimal improvement in GMM separation to continue training
    bad_epochs = 0
    best_state = None
    best_sep_avg = -float("inf")
    best_epoch = -1

    if epochs < warmup_epochs:
        print("Error: The number of epochs must be greater than or equal to the number of warmup epochs.")
        exit(1)

    device = torch.device('cuda')
    if device is None:
        print("Error: No GPU available. Please check your CUDA setup.")
        exit(1)

    # ------------ Load Data and Tokenizer ------------
    print(f"Loading training data from {train_file} and test data from {test_file}")
    train_data, test_data, noisy_mask, num_classes = load_data(train_file, train_data_path, dataset, noise_ratio, test_file=test_file, test_data_path=test_data_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    invert = (1 - noise_ratio) < (noise_ratio / (num_classes - 1))

    # ------------ Load Models ------------
    print(f"Loading BERT model from {bert_model}")
    torch.manual_seed(42)
    model1 = Bert(bert_model, head_type, dropout=0.0, num_classes=num_classes).to_device(device)
    torch.manual_seed(43)
    model2 = Bert(bert_model, head_type, dropout=0.0, num_classes=num_classes).to_device(device)
    
    # ------------ Load DataLoader ------------
    loader = DivideMixDataloader(
        batch_size=batch_size,
        tokenizer=tokenizer,
        num_workers=num_workers
    )

    # ------------ Load Optimizer and Loss ------------
    print("Setting up optimizers and losses...")
    semiloss = SemiLoss(lambda_u=lambda_u, rampup=rampup)
    optim1 = AdamW(
        [
            {"params": model1.bert.parameters(), "lr": learning_rate},
            {"params": model1.classifier.parameters(), "lr": learning_rate},
        ]
    )
    optim2 = AdamW(
        [
            {"params": model2.bert.parameters(), "lr": learning_rate},
            {"params": model2.classifier.parameters(), "lr": learning_rate},
        ]
    )
    per_sample_CEloss = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    negentropy = NegEntropy()
    decayed = False
    # decay_epoch = int(0.60 * epochs)

    # ------------ Load Pretrained Weights if available ------------
    start_epoch = 0
    warmup_done = False
    is_training_started = False
    prob1 = torch.zeros(len(train_data), device=device)
    prob2 = torch.zeros(len(train_data), device=device)
    # checkpoint_path = checkpoint_path.replace(".pth", f"_{noise_ratio}_{train_file.replace('_train.csv', '')}.pth")
    if os.path.exists(checkpoint_path):
        print("Found warm-up-completed models")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model1.load_state_dict(ckpt["model1"])
        model2.load_state_dict(ckpt["model2"])
        optim1.load_state_dict(ckpt["optim1"])
        optim2.load_state_dict(ckpt["optim2"])
        prob1 = ckpt["prob1"]
        prob2 = ckpt["prob2"]
        start_epoch = ckpt["epoch"]
        warmup_done = True
        is_training_started = True
    else:
        print("No warm-up-completed models. Training from scratch.")

    # ------------ Start Training ------------
    for epoch in range(start_epoch, epochs):
        # ---- Dropout Management ----
        # if not is_training_started and dropout_type == 'early':
        #     model1.dropout.p = p_early
        #     model2.dropout.p = p_early
        # elif is_training_started and dropout_type == 'late':
        #     model1.dropout.p = p_late
        #     model2.dropout.p = p_late
        # ---- Warmup Phase ----
        if not warmup_done and epoch < warmup_epochs:
            warmup_loader = loader.run(train_data, mode='warmup')
            print(f"Warmup training for Network 1")
            warmup_train(epoch, model1, optim1, warmup_loader, CEloss, negentropy, device)
            print(f"Warmup training for Network 2")
            warmup_train(epoch, model2, optim2, warmup_loader, CEloss, negentropy, device)
        else:
            # ---- Restore Best State ----
            if not is_training_started and best_state is not None:
                print(f"Restoring best state from epoch {best_epoch} with GMM separation {best_sep_avg:.4f}")
                model1.load_state_dict(best_state["model1"])
                model2.load_state_dict(best_state["model2"])
                optim1.load_state_dict(best_state["optim1"])
                optim2.load_state_dict(best_state["optim2"])
                prob1 = best_state["prob1"]
                prob2 = best_state["prob2"]
                is_training_started = True
            # ---- Training Phase ----
            pred1 = (prob1 > p_threshold) if not invert else (prob1 <= p_threshold) # predX.shape = [num_samples] (Boolean)
            pred2 = (prob2 > p_threshold) if not invert else (prob2 <= p_threshold) # True if the component with the lowest mean loss has probability higher than p_threshold
            if pred1.sum() == 0 or pred2.sum() == 0:
                print("Warning: no confident samples selected for this epoch.")
            # Wouldn't it be possible that no samples cross this threshold and everything gets classified as unlabelled?

            print(f"Training for Network 1")
            labelled_loader, unlabelled_loader = loader.run(train_data, mode='train', preds=pred2, probs=prob2)
            loss, Lx, Lu, penalty = train(epoch, model1, model2, optim1, semiloss, labelled_loader, unlabelled_loader, best_epoch, batch_size=batch_size, temperature=temperature, alpha=alpha, penalty_val=penalty_val, num_class=num_classes, device=device)
            print(f"Loss: {loss:.4f}, Lx: {Lx:.4f}, Lu: {Lu:.4f}, Penalty: {penalty:.4f}")
            print(f"Training for Network 2")
            labelled_loader, unlabelled_loader = loader.run(train_data, mode='train', preds=pred1, probs=prob1)
            loss, Lx, Lu, penalty = train(epoch, model2, model1, optim2, semiloss, labelled_loader, unlabelled_loader, best_epoch, batch_size=batch_size, temperature=temperature, alpha=alpha, penalty_val=penalty_val, num_class=num_classes, device=device)
            print(f"Loss: {loss:.4f}, Lx: {Lx:.4f}, Lu: {Lu:.4f}, Penalty: {penalty:.4f}")

        # ---- Evaluation Phase ----
        eval_loader = loader.run(train_data, mode='eval_train')
        print(f"Evaluating training data at epoch {epoch} for Model 1")
        prob1, losses1, gmm_sep1 = eval_train(model1, per_sample_CEloss, eval_loader, device=device)
        if dataset != 'reddit':
            save_orig_noisy_loss_histogram(noisy_mask, losses1, epoch, model=1)
        else:
            save_loss_histogram(losses1, epoch, model=1)
        print(f"Evaluating training data at epoch {epoch} for Model 2")
        prob2, losses2, gmm_sep2 = eval_train(model2, per_sample_CEloss, eval_loader, device=device)
        if dataset != 'reddit':
            save_orig_noisy_loss_histogram(noisy_mask, losses2, epoch, model=2)
        else:
            save_loss_histogram(losses2, epoch, model=2)

        # ---- GMM Mean Early Stopping During Warm-Up ----
        if not warmup_done:
            gmm_sep_avg = (gmm_sep1 + gmm_sep2) / 2
            print(f"Epoch {epoch} | sep1={gmm_sep1:.4f}, sep2={gmm_sep2:.4f}, avg={gmm_sep_avg:.4f}")
            if gmm_sep_avg > best_sep_avg + delta_sep_tol:
                best_sep_avg = gmm_sep_avg
                best_epoch = epoch
                bad_epochs = 0
                print(f"New best GMM separation found at epoch {epoch}: {best_sep_avg:.4f}")
                best_state = {
                    "model1": model1.state_dict(),
                    "model2": model2.state_dict(),
                    "optim1": optim1.state_dict(),
                    "optim2": optim2.state_dict(),
                    "prob1": prob1,
                    "prob2": prob2,
                    "epoch": epoch + 1
                }
                # torch.save(
                #     best_state,
                #     checkpoint_path,
                # )
            else:
                print(f"No significant improvement in GMM separation at epoch {epoch}. Current best: {best_sep_avg:.4f} at epoch {best_epoch}")
                bad_epochs += 1
                if bad_epochs >= gmm_patience:
                    print(f"Early stopping triggered at epoch {epoch} due to no significant GMM separation improvement for {gmm_patience} epochs.")
                    warmup_done = True

        # ---- Testing Phase ----
        print(f"Evaluating models at epoch {epoch}")
        test_loader = loader.run(test_data, mode='test')
        test_acc, test_f1, test_preds, test_labels, test_loss = test(model1, model2, test_loader)
        print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")

        # ---- Save Checkpoint ----
        checkpoint = {
            "model1": model1.state_dict(),
            "model2": model2.state_dict(),
            "optim1": optim1.state_dict(),
            "optim2": optim2.state_dict(),
            "prob1": prob1,
            "prob2": prob2,
            "epoch": epoch + 1,
        }
        # torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    # ---- Save Evaluation Results ----
    print("Saving evaluation results and predictions...")
    report = evaluate_model(test_preds, test_labels)
    save_evaluation(report, test_file, data_save_path, f'dividemix_{noise_ratio}')
    add_predictions_to_data(test_data, test_file, data_save_path, test_preds, f'dividemix_{noise_ratio}')