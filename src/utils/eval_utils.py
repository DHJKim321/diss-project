import json
from sklearn.metrics import classification_report

def evaluate_model(preds, labels):
    labels = [str(label) for label in labels]
    preds = [str(pred) for pred in preds]
    report = classification_report(labels, preds, output_dict=True)
    return report

def save_evaluation(evaluations, test_file, data_path):
    output_file = data_path + "metrics_" + test_file.replace("csv", "")
    with open(output_file, 'w') as f:
        json.dump(evaluations, f, indent=4)
    print(f"Evaluation results saved to {output_file}")