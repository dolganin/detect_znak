def calculate_metrics(predictions, targets, letters_list):
    metrics = {letter: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for letter in letters_list}

    for pred_seq, target_seq in zip(predictions, targets):
        for pred_char, target_char in zip(pred_seq, target_seq):
            if pred_char == target_char:
                metrics[pred_char]['TP'] += 1
                metrics[pred_char]['TN'] += len(letters_list) - 1  # Increment TN for all other letters
            else:
                metrics[pred_char]['FP'] += 1
                metrics[target_char]['FN'] += 1

    for letter in letters_list:
        TP = metrics[letter]['TP']
        FP = metrics[letter]['FP']
        TN = metrics[letter]['TN']
        FN = metrics[letter]['FN']
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        
        metrics[letter]['precision'] = precision
        metrics[letter]['recall'] = recall
        metrics[letter]['accuracy'] = accuracy

    return metrics


def average_metrics(metrics):
    avg_precision = sum(metric['precision'] for metric in metrics.values()) / len(metrics)
    avg_recall = sum(metric['recall'] for metric in metrics.values()) / len(metrics)
    avg_accuracy = sum(metric['accuracy'] for metric in metrics.values()) / len(metrics)

    return {'precision': avg_precision, 'recall': avg_recall, 'accuracy': avg_accuracy}



