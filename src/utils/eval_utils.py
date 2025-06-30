import json
from sklearn.metrics import classification_report
import pandas as pd

def evaluate_model(preds, labels):
    labels = [str(label) for label in labels]
    preds = [str(pred) for pred in preds]
    report = classification_report(labels, preds, output_dict=True, digits=4)
    print("Evaluation Report:")
    print(report)
    return report

def save_evaluation(evaluations, test_file, data_path, model_name):
    test_file = test_file.replace(".csv", "")
    output_file = f"{data_path}metrics_{test_file}_{model_name}.json"
    with open(output_file, 'w') as f:
        json.dump(evaluations, f, indent=4)
    print(f"Evaluation results saved to {output_file}")

def add_predictions_to_data(data, test_file, data_save_path, preds, model_name):
    test_file = test_file.replace(".csv", "")
    output_file = f"{data_save_path}evaluated_{test_file}_{model_name}.csv"
    data['predictions'] = preds
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def save_loss_as_df(epoch, losses, data_save_path):
    df = pd.DataFrame(columns=['model1_loss', 'model2_loss'])
    df['model1_loss'] = losses[0]
    df['model2_loss'] = losses[1]

    output_file = f"{data_save_path}dividemix_epoch_{epoch}_losses.csv"
    df.to_csv(output_file, index=False)