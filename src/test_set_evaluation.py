
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

"""
Functions to evaluation test set and create confusion matrix report
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

def test_set_prediction(model: torch.nn.Module,
                        test_dataloader: torch.utils.data.DataLoader,
                        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    """Used to evaluate model on test set dataloader

    Note: 
    - Assume binary class so use sigmoid in this function. If multiclass, modify the code to use softmax

    Args:
    model: A PyTorch model to used.
    test_dataloader: test set of dataloader.
    - You will notice that the code unpack 3 variables from the data loader: X,y and path
    - That is because my custom dataloader output those 3 variables (I added the path to the image)
    - If you use other dataloder which gives only X and y, modify the code so that it does not expect path (just delete path).

    Returns:
    y_true_tes: list of true label for each samples
    y_pred_test: list of predicted labels for each samples
    y_pred_prob_test: list of predicted probability using sigmoid function for each samples
    """
    
    print(f'Inferencing on {device}.')
    
    #initialize empty list
    y_true_test, y_pred_test, y_pred_prob_test = [], [], []

    model.eval()
    with torch.inference_mode(): 
        for X, y, path in tqdm(test_dataloader): #Loop through all batch in test_dataloader
            X, y = X.to(device), y.to(device)
            test_logits = model(X).squeeze()
            test_pred_prob = torch.sigmoid(test_logits)
            test_pred = torch.round(test_pred_prob)
            #To get y_pred, we convert logits to probability using sigmoid (use softmax in case of multiclass).
            #The probability is the probability of being positive class, or 1 in this case.
            #Then I just round it to get the label. We can modify the cutoff later if we want.

            y_true_test.extend(y.tolist()) #accumulate true label
            y_pred_test.extend(test_pred.tolist()) #accumulate pred label
            y_pred_prob_test.extend(test_pred_prob.tolist()) #accumulate predicted prob
                
            
    return y_true_test, y_pred_test, y_pred_prob_test


def create_report(y_true: list,
                  y_pred: list,
                  y_pred_prob: list = None,
                  class_map: dict = None):

    """Create report using SKlearn metrics

    Note: 
    - Output 3 metrics, accuracy, Balanced accuracy and F1 score.
    - If you want other metrics, modify the code accordingly. The trhee metrics do not use predicted probability
    - But for other metric like AUC you may need predicted probability as the input.

    Args:
    y_true: list of true label for each samples
    t_pred: list of predicted label for each samples
    y_pred_prob: list of predicted probability for each samples. Does not use it by default
    class_map: a dictionary which map each "encoded" class to the name. Eg: {0:"benign", 1: "malignant"}
    
    Returns:
     accuracy: accuracy as calculated by SKlearn.metric
     balanced_accuracy: balanced accuracy as calculated by SKlearn.metric
     f1: f1 as calculated by SKlearn.metric
    """
    
    #prediction on test set
    #create display labels
    if class_map:
        #If supplies class_map argument, create a new list from the dictionary.
        #It sort the label into a list accoring to the key. So value for key 0 will be the first item
        #Wich match how the target_names and display_labels argument expect to list of label to be (the first item in the list will be mapped to class 0, etc...)
        display_labels = [class_map[label] for label in sorted(class_map.keys())]
    else:
        display_labels = None


    confusion = confusion_matrix(y_true, y_pred)

    print(classification_report(y_true, y_pred, target_names= display_labels))
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = 'weighted')


    print('Confusion Matrix : \n', confusion)
    TN, FP, FN, TP = confusion.ravel()
    print('TN: ', TN, 'FP: ', FP, 'FN: ', FN, 'TP: ', TP)

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=display_labels, normalize = 'true')
    plt.show()

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Balanced accuracy: {balanced_accuracy:.2f}')
    print(f'F1: {f1:.2f}')


    return accuracy, balanced_accuracy, f1
 
 
 
    
