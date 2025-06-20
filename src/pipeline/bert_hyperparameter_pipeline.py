import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv

import torch
from src.utils.train_bert import train_one_run

if __name__ == "__main__":
    load_dotenv()
    train_file = os.getenv("TRAIN_FILE")
    train_data_path = os.getenv("TRAIN_DATA_PATH")
    bert_model = os.getenv("BERT_MODEL")
    model_save_dir = os.getenv("HYPERPARAM_MODEL_SAVE_PATH")

    use_dropout = os.getenv("USE_DROPOUT").lower() == "true"
    dropout = os.getenv("DROPOUT")
    epochs_list = list(map(int, os.getenv("EPOCHS_LIST").split(",")))
    batch_size_list = list(map(int, os.getenv("BATCH_SIZE").split(",")))
    learning_rate_list = list(map(float, os.getenv("LEARNING_RATE_LIST").split(",")))
    dropout_list = list(map(float, os.getenv("DROPOUT_LIST").split(",")))

    results = []
    best_f1 = -1
    best_report = None

    for epochs in epochs_list:
        for batch_size in batch_size_list:
            for learning_rate in learning_rate_list:
                print(f"Training with epochs={epochs}, batch_size={batch_size}, "
                        f"learning_rate={learning_rate}, dropout={dropout}")
                model_state, report = train_one_run(
                    csv_file=train_file,
                    csv_path=train_data_path,
                    bert_name=bert_model,
                    batch_size=int(batch_size),
                    lr=float(learning_rate),
                    epochs=int(epochs),
                    use_dropout=use_dropout,
                    dropout_p=float(dropout)
                )
                torch.cuda.empty_cache()
                
                results.append(report)
                if report['val_f1'] > best_f1:
                    best_f1 = report['val_f1']
                    best_report = report
                    best_model_state = model_state
    
    print("Best F1 Score:", best_f1)
    print("Best Report:", best_report)

# Save best configuration and model state
with open(os.path.join(model_save_dir, 'best_config.json'), 'w') as f:
    json.dump(best_report, f, indent=2)
with open(os.path.join(model_save_dir, 'best_model.pth'), 'wb') as f:
    torch.save(best_model_state, f)

# Save all results
with open(os.path.join(model_save_dir, 'hyperparameter_results.json'), 'w') as f:
    json.dump(results, f, indent=2)