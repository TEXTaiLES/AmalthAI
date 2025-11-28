import os
import csv

# tiny function to read score from a text file
def read_score(score_path):
    try:
        with open(score_path, 'r') as f:
            return float(f.read().strip())
    except:
        return None

# function to compare and save the best model run between different trials with 3 modes: Segmentation, Object Detection, Classification
def comparensave(experiment_dir, timestamp, csv_path, dataset_path, mode, maximize=True):
    best_score = None
    best_run = None
    if mode == "Seg":
        dataset_name = os.path.basename(dataset_path.rstrip('/'))
    elif mode == "OD":
        dataset_name = os.path.basename(os.path.dirname(dataset_path.rstrip('/')))
    elif mode == "Cls":
        dataset_name = os.path.basename(dataset_path.rstrip('/'))

    for model in os.listdir(experiment_dir):
        model_path = os.path.join(experiment_dir, model)
        if not os.path.isdir(model_path):
            continue
        
        # find the weights based on different file structures
        for run in os.listdir(model_path):
            run_path = os.path.join(model_path, run)
            config_path = os.path.join(run_path, 'config.json')
            score_path = os.path.join(run_path, 'result.txt')
            if mode == "Seg":
                weights_path = os.path.join(run_path, 'best_model.pth')
            elif mode == "OD":
                weights_path = os.path.join(run_path, 'weights', 'best.pt')
            elif mode == "Cls":
                weights_path = os.path.join(run_path, 'best_model.pth')
                config_path = os.path.join(run_path, 'class_names.json')

            if not os.path.isfile(score_path):
                continue

            score = read_score(score_path)
            if score is None:
                continue
            
            # locate the best score and return its details
            if (best_score is None or 
               (maximize and score > best_score) or 
               (not maximize and score < best_score)):
                best_score = score
                best_run = {
                    'timestamp': timestamp,
                    'dataset': dataset_name,
                    'model': model,
                    'run': run,
                    'score': score,
                    'weights': weights_path,
                    'config': config_path
                }
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=best_run.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(best_run)

    return best_run