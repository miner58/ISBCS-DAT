import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def calc_macro_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    accuracies = []
    for cls in classes:
        # 해당 클래스에 대한 True/False 레이블 생성
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)
        # 해당 클래스에 대한 정확도 계산
        acc = accuracy_score(y_true_cls, y_pred_cls)
        accuracies.append(acc)
    # 모든 클래스의 정확도의 평균 반환
    return np.mean(accuracies)

def calc_classification_report(y_true, y_pred, target_names=['stim_before', 'stim_after']):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return report

def calc_metrics(y_true, y_pred, prefix=None, target_names=['stim_before', 'stim_after']):
    if prefix is None:
        raise ValueError('prefix should be given : train, val, test')
    
    if not isinstance(y_true, np.ndarray):
        y_true = np.concatenate(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.concatenate(y_pred)

    report_dict = classification_report(y_true, y_pred, target_names=target_names,labels=[0,1], output_dict=True, zero_division=0)

    flat_report = {}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_key = f"{prefix}/report/{key}/{subkey}"
                # If subvalue is a float or int, log it directly
                if isinstance(subvalue, (float, int)):
                    flat_report[flat_key] = float(subvalue)
                else:
                    # Handle cases like support which might be integers
                    flat_report[flat_key] = subvalue
        else:
            flat_report[f"{prefix}/report/{key}"] = value
    flat_report[f"{prefix}/report/macro avg/accuracy"] = calc_macro_accuracy(y_true, y_pred)
    
    return flat_report