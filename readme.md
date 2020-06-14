# Caffe2CaptchaCracker

## To use this program, you need

1. generate your own dataset
2. training with the dataset that you generated


## To generate the dataset:

1. use command: ./genCaptchaCaffe.py digit_num sample_size db_name db_type db_path

This will generate a dataset contenting 10^<digit_num> classes of number and each number have <sample_size> of sample at <db_path> named <db_name>.<db_type> 


## To do training:
1. before training, make sure that the Params in trainCaptcha.py are correct:
    * MODEL_NAME: the name of saved model
    * TOTAL_ITERS: iteration number for training
    * CLASS_NUM: class number for each digit
    * BIT_NUM: digit number of the captcha image. (This param is rarely changed.)
    * USE_GPU: use gpu to train or not
    * TB_LEN: the length of progress bar
    * DATASET_NAME: the prefix of dataset name, mention that postfix would be either '_train.minidb' nor  '_test.minidb'
    * DATASET_FOLDER: the folder that the dataset were posed in
    * SAVE_MODEL: save the model during the training process or not
    * MODEL_SAVE_FOLDER: the folder for models to save in
    * LOAD_MODEL: use pretrained model for training or not
    * PRETRAIN_MODEL_PATH: the path to the pretrained model
2. python trainCaptcha.py
3. enjoy the result
