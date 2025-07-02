import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.optim import SGD
import torch.nn as nn
from src.model.bert import Bert
from src.utils.data_utils import load_full_data, load_test_data
from src.utils.eval_utils import save_evaluation, add_predictions_to_data, evaluate_model, save_loss_as_df
from src.utils.dividemix_utils import warmup_train, train, eval_train, test
from src.data.DivideMixDataloader import DivideMixDataloader
from src.modules.losses import SemiLoss, NegEntropy
from transformers import BertTokenizer

from dotenv import load_dotenv

import random
torch.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
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

    # ---- Training Variables ----
    bert_model = os.getenv("BERT_MODEL")
    batch_size = int(os.getenv("BATCH_SIZE"))
    learning_rate = float(os.getenv("LEARNING_RATE"))
    warmup_epochs = int(os.getenv("WARMUP_EPOCHS"))
    epochs = int(os.getenv("EPOCHS"))
    alpha = float(os.getenv("ALPHA"))
    lambda_u = float(os.getenv("LAMBDA_U"))
    p_threshold = float(os.getenv("P_THRESHOLD"))
    temperature = float(os.getenv("SHARPENING_TEMPERATURE"))
    num_workers = int(os.getenv("NUM_WORKERS"))
    momentum = float(os.getenv("MOMENTUM"))
    weight_decay = float(os.getenv("WEIGHT_DECAY"))
    augmentation = os.getenv("AUGMENTATION")
    head_type = os.getenv("HEAD_TYPE")

    if epochs < warmup_epochs:
        print("Error: The number of epochs must be greater than or equal to the number of warmup epochs.")
        exit(1)

    device = torch.device('cuda')
    if device is None:
        print("Error: No GPU available. Please check your CUDA setup.")
        exit(1)

    # ------------ Load Data and Tokenizer ------------
    print("Loading training and testing data...")
    train_data = load_full_data(train_file, train_data_path)
    test_data = load_test_data(test_file, test_data_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # ------------ Load Models ------------
    print(f"Loading BERT model from {bert_model}")
    model1 = Bert(bert_model, head_type).to_device(device)
    model2 = Bert(bert_model, head_type).to_device(device)

    # ------------ Load DataLoader ------------
    loader = DivideMixDataloader(
        batch_size=batch_size,
        tokenizer=tokenizer,
        num_workers=num_workers
    )

    # ------------ Load Optimizer and Loss ------------
    print("Setting up optimizers and losses...")
    semiloss = SemiLoss(lambda_u=lambda_u)
    optim1 = SGD(model1.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optim2 = SGD(model2.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    per_sample_CEloss = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    negentropy = NegEntropy()

    all_loss = [[], []] # Store losses for model1 and model2

    # ------------ Start Training ------------
    for epoch in range(epochs):
        lr = learning_rate
    # ---- Learning Rate Decay ----
        if epoch >= 40:
            lr /= 10 #  Reduce learning rate after 40 epochs
        for param_group in optim1.param_groups:
            param_group['lr'] = lr
        for param_group in optim2.param_groups:
            param_group['lr'] = lr

        if epoch < warmup_epochs:
            # ---- Warmup Phase ----
            warmup_loader = loader.run(train_data, mode='warmup')
            print(f"Warmup training for Network 1")
            warmup_train(epoch, model1, optim1, warmup_loader, CEloss, negentropy, device)
            print(f"Warmup training for Network 2")
            warmup_train(epoch, model2, optim2, warmup_loader, CEloss, negentropy, device)
        else:
            # ---- Training Phase ----
            pred1 = (prob1 > p_threshold) # predX.shape = [num_samples] (Boolean)
            pred2 = (prob2 > p_threshold) # True if the component with the lowest mean loss has probability higher than p_threshold
            if pred1.sum() == 0 or pred2.sum() == 0:
                print("Warning: no confident samples selected for this epoch.")
            # Wouldn't it be possible that no samples cross this threshold and everything gets classified as unlabelled?

            print(f"Training for Network 1")
            labelled_loader, unlabelled_loader = loader.run(train_data, mode='train', preds=pred1, probs=prob1)
            train(epoch, model1, model2, optim1, semiloss, labelled_loader, unlabelled_loader, warmup_epochs, batch_size=batch_size, temperature=temperature, alpha=alpha, device=device)
            print(f"Training for Network 2")
            labelled_loader, unlabelled_loader = loader.run(train_data, mode='train', preds=pred2, probs=prob2)
            train(epoch, model2, model1, optim2, semiloss, labelled_loader, unlabelled_loader, warmup_epochs, batch_size=batch_size, temperature=temperature, alpha=alpha, device=device)

        # ---- Testing Phase ----
        print(f"Evaluating models at epoch {epoch}")
        test_loader = loader.run(test_data, mode='test')
        test_acc, test_f1, test_preds, test_labels = test(model1, model2, test_loader)
        print(f"Epoch: {epoch}, Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")

        # ---- Evaluation Phase ----
        print(f"Evaluating training data at epoch {epoch}")
        eval_loader = loader.run(train_data, mode='eval_train')
        prob1, all_loss[0] = eval_train(model1, all_loss[0], per_sample_CEloss, eval_loader, device=device)
        prob2, all_loss[1] = eval_train(model2, all_loss[1], per_sample_CEloss, eval_loader, device=device)
        save_loss_as_df(epoch, all_loss, checkpoint_path)

    # ---- Save Evaluation Results ----
    print("Saving evaluation results and predictions...")
    report = evaluate_model(test_preds, test_labels)
    save_evaluation(report, test_file, data_save_path, 'dividemix')
    add_predictions_to_data(test_data, test_file, data_save_path, test_preds, 'dividemix')