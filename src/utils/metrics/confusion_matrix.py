import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def array_to_text(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()  # numpy array를 list로 변환
    text = ','.join(map(str, arr))  # 리스트를 ','로 구분된 문자열로 변환
    return text

# 텍스트를 numpy array로 변환
def text_to_array(text):
    arr = np.array(list(map(float, text.split(','))))  # ','로 분할 후 float 타입으로 변환하여 numpy array 생성
    return arr

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def calc_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred), normalize='true')
    return cm
    