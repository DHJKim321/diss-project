import json

def calc_accuracy(preds, labels):
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) * 100 if labels else 0.0

def calc_precision(preds, labels):
    true_positives = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    predicted_positives = sum(p == 1 for p in preds)
    return true_positives / predicted_positives * 100 if predicted_positives > 0 else 0.0

def calc_recall(preds, labels):
    true_positives = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    actual_positives = sum(l == 1 for l in labels)
    return true_positives / actual_positives * 100 if actual_positives > 0 else 0.0

def calc_f1(preds, labels):
    precision = calc_precision(preds, labels)
    recall = calc_recall(preds, labels)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model(preds, labels):
    accuracy = calc_accuracy(preds, labels)
    precision = calc_precision(preds, labels)
    recall = calc_recall(preds, labels)
    f1_score = calc_f1(preds, labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def save_evaluation(evaluations, file_name, data_path):
    output_file = data_path + "evaluated_" + file_name
    with open(output_file, 'w') as f:
        json.dump(evaluations, f, indent=4)
    print(f"Evaluation results saved to {output_file}")