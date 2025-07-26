import os, sys
import nltk
nltk.download('averaged_perceptron_tagger', quiet=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv
from src.utils.mixtext_utils import *
from src.utils.data_utils import get_mixtext_data_idx
from src.data.dataloader.MixTextDataloader import MixTextDataLoader
from src.model.mixtextbert import MixTextBert
from src.modules.mixtext_loss import SemiLoss
from transformers import BertTokenizer
import torch
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn


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
    warmup_checkpoint_path = os.getenv("MIXTEXT_CHECKPOINT_PATH")
    n_labelled_per_class = int(os.getenv("N_LABELLED_PER_CLASS"))
    pickle_path = os.getenv("PICKLE_PATH")
    # Model Training Variables
    bert_model = os.getenv("BERT_MODEL")
    batch_size_x = int(os.getenv("BATCH_SIZE_X"))
    batch_size_u = int(os.getenv("BATCH_SIZE_U"))
    learning_rate = float(os.getenv("LEARNING_RATE"))
    warmup_epochs = int(os.getenv("WARMUP_EPOCHS"))
    epochs = int(os.getenv("EPOCHS"))
    head_type = os.getenv("HEAD_TYPE")
    num_workers = int(os.getenv("NUM_WORKERS"))
    dropout = float(os.getenv("DROPOUT"))
    lambda_u = float(os.getenv("LAMBDA_U"))
    rampup = int(os.getenv("RAMPUP"))
    T = float(os.getenv("SHARPENING_TEMPERATURE"))
    alpha = float(os.getenv("ALPHA"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------ Load Data and Tokenizer ------------
    print(f"Loading training data from {train_file} and test data from {test_file}")
    train_data, test_data, train_labelled_idxs, train_unlabelled_idxs, val_idxs, num_classes = get_mixtext_data_idx(train_file, train_data_path, test_file, test_data_path, n_labelled_per_class)
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # ------------ Load DataLoader ------------
    print("Loading MixText DataLoader")
    loader = MixTextDataLoader(
        batch_size_x=batch_size_x,
        batch_size_u=batch_size_u,
        tokenizer=tokenizer,
        num_workers=num_workers,
        pickle_path=pickle_path
    )
    labelled_loader = loader.run(
        data=train_data.iloc[train_labelled_idxs].reset_index(drop=True),
        mode='labelled',
    )
    unlabelled_loader = loader.run(
        data=train_data.iloc[train_unlabelled_idxs].reset_index(drop=True),
        mode='unlabelled',
    )
    val_loader = loader.run(
        data=train_data.iloc[val_idxs].reset_index(drop=True),
        mode='val'
    )
    test_loader = loader.run(
        data=test_data,
        mode='test'
    )
    print(f"Train labelled data size: {len(labelled_loader.dataset)}")
    print(f"Train unlabelled data size: {len(unlabelled_loader.dataset)}")
    print(f"Validation data size: {len(val_loader.dataset)}")
    print(f"Test data size: {len(test_loader.dataset)}")

    # ------------ Load Models ------------
    print(f"Loading BERT model from {bert_model}")
    model = MixTextBert(
        bert_model=bert_model,
        head_type=head_type,
        num_classes=num_classes,
        use_dropout=True,
        dropout=dropout
    ).to(device)

    # ------------ Load Optimizer ------------
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": learning_rate},
            {"params": model.classifier.parameters(), "lr": learning_rate * 100},
        ])
    semiloss = SemiLoss(
        lambda_u = lambda_u,
        rampup=rampup
    )
    criterion = nn.CrossEntropyLoss()

    # ------------ Start Training ------------
    best_acc = 0
    print("Starting training...")
    for epoch in range(epochs):
        tqdm.write(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train(labelled_loader, unlabelled_loader, model, optimizer, semiloss, epoch, num_classes, T, alpha, device)
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

        val_loss, val_acc = validate(val_loader, model, criterion)
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            tqdm.write(f"Validation accuracy improved to {best_acc:.4f}, saving model...")
            torch.save(model.state_dict(), os.path.join(model_save_path, f"mixtext_bert_{dataset}_best.pth"))
            tqdm.write(f"Testing best model on test set...")
            test_loss, test_acc = validate(test_loader, model, criterion)
            tqdm.write(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    tqdm.write("Training complete.")
    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_save_path, f"mixtext_bert_{dataset}_final.pth"))
    tqdm.write(f"Final model saved to {os.path.join(model_save_path, f'mixtext_bert_{dataset}_final.pth')}")