
"""
Custom train and val loop
"""

import torch
from save_model import save_model
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt


def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                target_dir: str,
                model_name: str,
                epochs: int = 50,
                device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):


    """Full training/validate loop for a pytorch model and save model with the least loss

    Note: 
    - This function assume the use of BCElosswithlogits, hence logits is passed directly to the loss_function.
    - If you want to use another loss function, you need to modify the training code
    - Does not utilize early stopping, only early save
    - Automatically save model as .pth. It saves state_dict, epoch, best_val_loss and optimizer_state_dict
    - The metrics used is balanced accuracy and F1 from scikitlearn.metrics. 
    - Depends on other custom function, save_model, found in utils_save_model.py
    - Automatically plot loss and metrics using matplotlib
    - Device is prefebly setup in the beginning of the code in the device agnostic code cell
    

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    val_dataloader: A DataLoader instance for the model to be validated on.
    - You will notice that the code unpack 3 variables from the data loader: X,y and path
    - That is because my custom dataloader output those 3 variables (I added the path to the image)
    - If you use other dataloder which gives only X and y, modify the code so that it does not expect path (just delete path).
    
    epochs: An integer indicating how many epochs to train for.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    target_dir: target directory name to save the best model with the least loss
    model_name: Name of the model. Do not include .pth in the name.
    

  Returns:
    A dictionary of training and testing values for later plotting
    In the form: {train_loss_values: [...],
                  val_loss_values: [...],
                  etc...} 
    """
    
    print(f'Training on {device}.')
    # Create empty loss lists to track values
    
    train_loss_values = []
    val_loss_values = []
    epoch_count = []
    
    train_accuracy_values = []
    train_balanced_accuracy_values = []
    train_f1_values = []
    
    val_accuracy_values = []
    val_balanced_accuracy_values = []
    val_f1_values = []

    #initialize best lost
    val_loss_best = float('inf') #initialize best loss to inifite

    # Create training and testing loop
    for epoch in tqdm(range(epochs)): #use tqdm for progressbar
        print(f"Epoch: {epoch}\n-------") #print epoch
        ### Training
        train_loss = 0 #default train loss for that epoch.
        y_true_train, y_pred_train = [], []  # Initialize empty lists for true and predicted labels
        y_true_val, y_pred_val = [], []  # Initialize empty lists for true and predicted labels
        
        #This is just for tracking purpose
        
        # Add a loop to loop through training mni batches
        for batch, (X, y, path) in enumerate(train_dataloader): 
            #loop throuugh all minibatch in an epoch
            #We actually do not need to enumerate the dataloader and define batch
            #It can be just for X, y, path in train_dataloader: 
            #But batch can be used to keep track of progress of batch in each epoch so I just leave it in case you want to track the number of batch done
            #Also, since my dataloader output 3 things: tensor of image, label and path to image, i assign path variable too but it isn't actually used
             
             
            X, y = X.to(device), y.to(device)
            model.train()  #set mode
            # 1. Forward pass
            #print(X)
            #print(X.max())
            y_logits = model(X).squeeze() #pass ALL image in the 32 batches.
            #print(y_logits.max())
            y_pred = torch.round(torch.sigmoid(y_logits)) 
            #To get y_pred, we convert logits to probability using sigmoid (use softmax in case of multiclass).
            #Then I just round it to get the label.
             
            #This is not needed for training since we use BCE with LogitLoss which can directly use logits. However, we do it to calculate metrics.
            
            #print(torch.sigmoid(y_logits))
            #print(y_pred)
            #print(y)
            
            y_true_train.extend(y.tolist()) #accumulate true label
            y_pred_train.extend(y_pred.tolist()) #accumulate pred label
            
            # print("y_true")
            # print(y_true_train)
            # print("y_pred")
            # print(y_pred_train)
            
            # 2. Calculate loss (per mini batch)
            #minibatch gradient descent calculate the loss and optimize the weight per mini batches 
            #as opposed to calculating and optjimze after all data has bee seen in an epoch
            loss = loss_fn(y_logits, y.type(torch.float32)) #Because we use BCE loss, we can directly pass logits
            train_loss += loss # accumulatively add up the loss from ech minibatches fo tracking purpose

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step() #optimize network per mini batches batch


        
        #Find average train loss per minibatch
        train_loss /= len(train_dataloader)
        
        #Calculate metrics for that epoch using the accumulated label
        train_accuracy = accuracy_score(y_true_train, y_pred_train) 
        train_balanced_accuracy = balanced_accuracy_score(y_true_train, y_pred_train) 
        train_f1_score = f1_score(y_true_train, y_pred_train)        

        
        ### val
        # Setup variables for accumulatively adding up loss and accuracy for that epoch
        val_loss = 0
        model.eval()
        with torch.inference_mode(): #context management
            for X, y, path in val_dataloader: #Loop through all mini batch in the test set. We don't use enumerate for this because
                #we don't care to track what test minibatch the progress is at.
                X, y = X.to(device), y.to(device)
                val_logits = model(X).squeeze()
                val_pred = torch.round(torch.sigmoid(val_logits))
            
            
                # 2. Calculate loss (accumatively)
                val_loss += loss_fn(val_logits, y.type(torch.float32)) # accumulatively add up the loss per mini batch for that epoch

                # 3. accumulate label for metric calculation
                # y_true_val = y.tolist()
                # y_pred_val = val_pred.tolist()
                
                y_true_val.extend(y.tolist()) #accumulate true label
                y_pred_val.extend(val_pred.tolist()) #accumulate pred label


                
            
            # Calculations on test metrics need to happen inside torch.inference_mode()
            # Divide total test loss by length of test dataloader (per batch)
            val_loss /= len(val_dataloader)
            
            #Now you get the average test loss per minibatch for that epoch

            # For metric
            #Calculate metric based on accumulate true and pred label
            val_accuracy = accuracy_score(y_true_val, y_pred_val)
            val_balanced_accuracy = balanced_accuracy_score(y_true_val, y_pred_val)
            val_f1_score = f1_score(y_true_val, y_pred_val)
            
            #for save model
            #check if the loss is less than the best val loss
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                best_epoch = epoch
                best_checkpoint = {
                    'epoch': epoch,
                    'best_val_loss': val_loss_best,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                
            
        #Printing of statistic every x epoch. Default to every epoch.    
        if epoch % 1 == 0:
            
            epoch_count.append(epoch) #append the number of epoch to be used as X axis to plot loss
            
            train_loss_values.append(train_loss.item()) #append data for train loss to plot
            
            val_loss_values.append(val_loss.item()) #append data for val loss to plot
            
            #append data for metric to plot
            train_accuracy_values.append(train_accuracy)
            train_balanced_accuracy_values.append(train_balanced_accuracy)
            train_f1_values.append(train_f1_score) 
            
            val_accuracy_values.append(val_accuracy)
            val_balanced_accuracy_values.append(val_balanced_accuracy)
            val_f1_values.append(val_f1_score)
            

                       
        ## Print out what's happening
        print(f"Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}")
        print(f"Train accuracy: {train_accuracy:.5f} | Train balanced accuracy: {train_balanced_accuracy:.5f} | Train F1 score: {train_f1_score:.5f}")    
        print(f"Validation accuracy: {val_accuracy:.5f} | Validation balanced accuracy: {val_balanced_accuracy:.5f} | Validation F1 score: {val_f1_score:.5f}")
        
    #Gather results into dictionary
    results = {'epochs': epoch_count,
               'best_epoch': best_epoch,
               'train_loss_values': train_loss_values,
               'val_loss_values': val_loss_values,
               'train_accuracy_values': train_accuracy_values,
               'val_accuracy_values': val_accuracy_values,
               'train_balanced_accuracy_values': train_balanced_accuracy_values,
               'val_balanced_accuracy_values': val_balanced_accuracy_values,
               'train_f1_values': train_f1_values,
               'val_f1_values': val_f1_values               
    }    
    
    #Saving model with the best epoch.
    print(f'The best model is at epoch {best_epoch}')
    print('Saving model state_dict, optimizer, best_val_loss, epoch')
    save_model(best_checkpoint, target_dir=target_dir, model_name=f'{model_name}_epoch_{best_epoch}.pth')
    
    # Plot the loss curves
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Plot the loss curves on the first subplot
    axs[0].plot(results['epochs'], results['train_loss_values'], label="Train loss")
    axs[0].plot(results['epochs'], results['val_loss_values'], label="Val loss")
    axs[0].axvline(results['best_epoch'], linestyle='--', color='r',label='Best Epoch')
    axs[0].set_title("Training/Val loss curves")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot the metric curves on the second subplot
    axs[1].plot(results['epochs'], results['train_accuracy_values'], label="Train Accuracy")    
    axs[1].plot(results['epochs'], results['train_balanced_accuracy_values'], label="Train Balanced Accuracy")
    axs[1].plot(results['epochs'], results['train_f1_values'], label="Train F1")
    axs[1].plot(results['epochs'], results['val_accuracy_values'], label="Val Accuracy")    
    axs[1].plot(results['epochs'], results['val_balanced_accuracy_values'], label="Val Balanced Accuracy")
    axs[1].plot(results['epochs'], results['val_f1_values'], label="Val F1")
    axs[1].axvline(results['best_epoch'], linestyle='--', color='r',label='Best Epoch')
    axs[1].set_title("Train/Val metric")
    axs[1].set_ylabel("Score")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    return results
