# softcomputing

1. Run `resize.py` file to get unified images from dataset

2. In `multiplication.py` file
   * set `rotation_range` in line 7 [Other options](https://keras.io/api/preprocessing/image/#imagedatagenerator-class)
   * set `range` from 0 to 50 in line 16 to specify how many classes you want to multiplicate
   * specify `i` variable in line 31 to set how many copies should be given for each image from dataset then run it

3. In `cnnNetworkTransfer.py`:
   * choose suitable pre-trained model - line 61 [Available models](https://keras.io/api/applications/#available-models)
   * choose optimizer and set learning rate - line 78 [Optimizers](https://keras.io/api/optimizers/#available-optimizers)
   * set batch size and number of epochs - lines 82, 83