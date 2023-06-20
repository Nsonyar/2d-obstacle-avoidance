# ss20_03_training_model
At this step, training of the model is taking place. To enjoy GPU support, 
the container is properly set up and if run with the --gpu flag, ready for
training with gpu acceleration. 

Once the following command is run from within the scripts folder, it will ask 
the user to select a dataset for training, to create a model and to have it saved in 
a subfolder and named with its corresponding identifier. The identifier relates 
bagfiles, datasets and created models, in order to keep track of different tests 
and runs.
```bash
python3 training_model.py
```
The script can also be launched with parameters as follows.
```bash
python3 training_model.py [-h] [-n NAME] [-e EPOCHS] [-s STEPS]
                         [-bs BATCH_SIZE] [-lr LEARNING_RATE] [-tv RATIO]
                         [-m MODE]
```
- **[-h]** -> help
- **[-n NAME]** -> Name of model
- **[-e EPOCHS]** -> No. of epochs default: 15
- **[-s STEPS]** -> No. of steps, default: 1000
- **[-bs BATCH_SIZE]** -> Batch size, default: 64
- **[-lr LEARNING_RATE]** -> Learning rate, default: 0.0002
- **[-tv RATIO]** -> Training vs Validation ratio, default: 0.8
- **[-m MODE]** -> Currently not used