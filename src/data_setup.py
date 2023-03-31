"""
Contains functions for setting up dataloder 
"""
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple, Dict, List
import pandas as pd


# 1. Subclass torch.utils.data.Dataset
class my_dataset(Dataset):
    
    # 2. Initialize with a targ_dir, transform (optional), class mapping and dataframe for path and label.
    def __init__(self, targ_dir: str,  class_map: dict, label_df, transform=None,) -> None:
        
        # 3. Create class attributes
        
        
        # Firstly, Get all image paths
        #self.paths = list(pathlib.Path(targ_dir).glob("*.jpg")) #old code, did not filter image with non-matchig label
        
        #code for filter the paths to keep only the ones with labels in the DataFrame
        all_paths = list(pathlib.Path(targ_dir).glob("*.jpg"))

        self.paths = []
        for path in all_paths:
            img_path = os.path.relpath(str(path), '../').replace('\\', '/').replace('data/', '')
            if img_path in label_df['image'].values:
                self.paths.append(path)
        #now we get all the path that has labels in the text files
        # pathlib.Path(train_dir): The pathlib.Path() function is called with the train_dir variable as its argument. This creates a Path object representing the file system path to the directory.
        # pathlib.Path(train_dir).glob("*.jpg"): The glob() method is called on the created Path object, with the pattern "*.jpg" as its argument. 
        # #The glob() method searches for files and directories that match the given pattern. 
        # #In this case, the pattern "*.jpg" is used to find all files with the .jpg extension (JPEG files) in the train_dir directory.
        # list(pathlib.Path(train_dir).glob("*.jpg")): The glob() method returns a generator object, which can be iterated over to access the matched files. 
        # By passing this generator to the list() function, a list is created that contains all the JPEG files found in the train_dir directory.
        # In summary, this code snippet creates a list of all JPEG files in the directory specified by the train_dir variable using the pathlib module's Path class and its glob() method

        # Setup transforms
        self.transform = transform
        
        # Class mapping
        self.label_to_name = class_map 
        
        #All class label
        self.classes = list(class_map.keys())
               
        #path and label df
        self.path_label_df = label_df

    # 4. Overwrite the __len__() method to get the amount of all samples
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths) #Count all the path in the self.paths, which we filtered to the matching rows only 
    
    # 5. Overwrite the __getitem__() method 
    # It will take intex. It must return at least the sample and label.
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        "Returns one sample of data (data and label and path to the data) (X, y, path)."
        
        img = Image.open(self.paths[index])  #open the image in the given index using list which sef.paths defined in the init.
        img_path = os.path.relpath(str(self.paths[index]), '../').replace('\\', '/').replace('data/', '') 
        #get the relative path and convert the iamge path to the same format as in the txt files
        #relpath finds the relative path
        #replace \\ with / and data/ with blank to makes it match the format in the txt file
        
        #For debug, raise error if no matching rows
        #matching_rows = self.path_label_df.loc[self.path_label_df.image == img_path]

        #if matching_rows.empty:
            #raise ValueError(f"Image path {img_path} not found in {self.path_label_df}.")
        
        class_label = int(self.path_label_df.loc[self.path_label_df.image == img_path, 'label'])
        # get the class label. Find the row which match the given path, get the label from the label column and convert to integer.
        #You can also use .values[0] or .item() to get the scalar value from panda dataframe

        #convert it to class ID. using the dictionary created with the find_classes function.
        class_name= self.label_to_name[class_label] 
        
        #self.label_to_name is a dictionary, which we specify the key as the class_label, which will return the value, which is the class name.

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_label, img_path  # return data, label, path to image
        else:
            return img, class_label,img_path # return data, label, path to image 


#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0 #Somehow windows has problem with multi-process so > 0 does not works. IDK why.


def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    val_dir:str,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    test_transform:transforms.Compose,
    train_label_df,
    val_label_df,
    test_label_df,
    class_map:dict,
    batch_size: int, 
    num_workers: int=NUM_WORKERS):
    
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    val_dir: path to validation directory.
    test_dir: Path to testing directory.
    train,val,test_transform: torchvision transforms to perform on training and testing data.
    train,val,test_label_df: dataframe which contrain the path to image in the column 'image' and the corresponding label in the column 'image' as defined in my_dataset class.
    If the label and data is different, you need to modify how to get data and label on the my_dataset class.
    class_map: dictionary which map encoded class to name eg. {0:'Benign', 1:'Malignant'}
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    train_data: custom dataset using my_dataset class
    val_data: custom dataset using my_dataset class
    test_data: custom dataset using my_dataset class
    train_dataloader: custom data loader generated from train_data
    val_dataloade: custom data loader generated from val_data
    test_dataloader: custom data loader generated from test_data
    class_label_all: all unique class label

    Example usage:
        #Setup directory
        train_dir = '../data/train/'
        val_dir = '../data/val/'
        test_dir = '../data/test/'

        #set up class mapping
        classes = [0, 1]  #  class label
        label_to_name = {0:"benign", 1: "malignant"}  # Mapping 

        transforms = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.ToTensor()])
            
        import data_setup
        train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader, class_labels = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                    val_dir = val_dir,                
                                                                                    test_dir= test_dir,
                                                                                    train_transform=transforms, 
                                                                                    val_transform=val_transforms, 
                                                                                    test_transform=test_transforms,
                                                                                    train_label_df = data_label_train,
                                                                                    val_label_df = data_label_val,
                                                                                    test_label_df = data_label_test,
                                                                                    class_map = label_to_name,                
                                                                                    batch_size=32) 

  """
  # Use create dataset(s)
  train_data = my_dataset(targ_dir=train_dir, 
                                      transform=train_transform, 
                                      class_map = class_map, 
                                      label_df = train_label_df)
  val_data = my_dataset(targ_dir=val_dir, 
                                     transform=val_transform,
                                     class_map = class_map, 
                                     label_df = val_label_df)                                    
  test_data = my_dataset(targ_dir=test_dir, 
                                     transform=test_transform,
                                     class_map = class_map,
                                     label_df = test_label_df)
                                     
                                     

  # Get class names
  class_label_all = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last= True, 
      pin_memory=True,
  )
  
  val_dataloader = DataLoader(
      val_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )  
  
  
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )
  

  return train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader, class_label_all
