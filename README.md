# Breast-Density-Classification-MONAI

* Install Dependencies
pip install -t req.txt

* Run the file monai.py with the following arguments:

  * -r: path to checkpoint from where training needs to be resumed.
  * -m : Name of the model architecture. Choose amongst - resnet, vgg19, vgg16, wide_resnet50_2 and densenet
  * -d: Name of the breast density dataset. Choose amongst - dmist2, dmist3, dmist4, mgh and dmist
  * -s : seed
  * -m : Whether to train using conventional pytorch or using monai trainer
  * -p: Percentage of data from the mentioned dataset on which the model has to be trained
  
