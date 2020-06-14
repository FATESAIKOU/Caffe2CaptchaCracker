from matplotlib import pyplot

from IPython import display
from PIL import Image

import numpy as np
import os
import shutil
import time

import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils

from caffe2.python import core, net_drawer, workspace, cnn, optimizer, brew, model_helper, visualize
from caffe2.python.predictor_constants import predictor_constants as predictor_constants


MODEL_NAME = 'captcha'
TOTAL_ITERS = 20000
CLASS_NUM = 10
BIT_NUM = 4
USE_GPU = True
TB_LEN = 50

DATASET_NAME = 'final2_captcha'
DATASET_FOLDER = '/home/fatesaikou/testML/machine_learning/crackCaptcha'

SAVE_MODEL = False
MODEL_SAVE_FOLDER = '/home/fatesaikou/testML/machine_learning/crackCaptcha/models'

LOAD_MODEL = False
PRETRAIN_MODEL_PATH = '/home/fatesaikou/testML/machine_learning/crackCaptcha/models/captcha_train_40000.mdl'


""" Layer Defination """

def AddInput(model, batch_size, db, db_type):
    data_uint8, label1, label2, label3, label4  = model.TensorProtosDBInput(
        [], ["data_uint8", "label1", "label2", "label3", "label4"], batch_size=batch_size,
        db=db, db_type=db_type)
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    data = model.Scale(data, data, scale=float(1./256))
    data = model.StopGradient(data, data)
    data, _ = model.Reshape(data, ['data', '_'], shape=[-1, 1, 60, 160])

    return data, [label1, label2, label3, label4]

def AddNetModel(model, data):
    conv1 = model.Conv(data, 'conv1', dim_in=1, dim_out=16, kernel=5)
    pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
    conv2 = model.Conv(pool1, 'conv2', dim_in=16, dim_out=32, kernel=5)
    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
    
    fc3 = model.FC(pool2, 'fc3', dim_in=32 * 12 * 37, dim_out=512)
    fc3 = model.Relu(fc3, fc3)
    fc4 = model.FC(fc3, 'fc4', 512, BIT_NUM * CLASS_NUM)
    fc4s = model.DepthSplit(fc4, BIT_NUM, split=CLASS_NUM, axis=1)

    softmax = []
    softmax.append(model.Softmax(fc4s[0], 'softmax0'))
    softmax.append(model.Softmax(fc4s[1], 'softmax1'))
    softmax.append(model.Softmax(fc4s[2], 'softmax2'))
    softmax.append(model.Softmax(fc4s[3], 'softmax3'))
    
    return softmax

def AddAccuracy(model, softmax, label):
    accuracy = []
    accuracy.append(model.Accuracy([softmax[0], label[0]], "accuracy0"))
    accuracy.append(model.Accuracy([softmax[1], label[1]], "accuracy1"))
    accuracy.append(model.Accuracy([softmax[2], label[2]], "accuracy2"))
    accuracy.append(model.Accuracy([softmax[3], label[3]], "accuracy3"))

    return accuracy

def AddTrainingOperators(model, softmax, label):
    # Caculate CrossEntropy
    xents = []
    xents.append(model.LabelCrossEntropy([softmax[0], label[0]], 'xent0'))
    xents.append(model.LabelCrossEntropy([softmax[1], label[1]], 'xent1'))
    xents.append(model.LabelCrossEntropy([softmax[2], label[2]], 'xent2'))
    xents.append(model.LabelCrossEntropy([softmax[3], label[3]], 'xent3'))

    # Join Xents
    xent, _ = model.net.Concat(xents, outputs=2, axis=0)

    # Caculate Loss
    loss = model.AveragedLoss(xent, "loss")

    # Caculate Accuracy
    AddAccuracy(model, softmax, label)

    # Define Learing Rate and some other params.
    ITER = model.Iter("iter")
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.01, policy="step", stepsize=5000, gamma=0.999)
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
 
    # Do gradient
    gradient_map = model.AddGradientOperators([loss])
    for param in model.params:
        param_grad = gradient_map[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)


""" Load/Store Model Defination """

def GetCheckpointParams(train_model):
    params = [str(p) for p in train_model.GetParams()]
    params.extend([str(p) for p in train_model.GetComputedParams()])
    assert len(params) > 0

    return params

def SaveModel(train_model, save_dir, epoch):
    predictor_export_meta = pred_exp.PredictorExportMeta(
        predict_net=train_model.net.Proto(),
        parameters=GetCheckpointParams(train_model),
        inputs=['data_uint8'],
        outputs=['softmax'],
        shapes={
            'data': {1, 9600},
            'softmax': {1, 40}
        }
    )
    
    model_path = '%s/%s_%d.mdl' % (
        save_dir,
        train_model.net.Proto().name,
        epoch,
    )

    pred_exp.save_to_db(
        db_type='minidb',
        db_destination=model_path,
        predictor_export_meta=predictor_export_meta,
    )

def LoadModel(path):
    meta_net_def = pred_exp.load_from_db(path, 'minidb')
    init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.GLOBAL_INIT_NET_TYPE))
    predict_init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.PREDICT_INIT_NET_TYPE))
    
    predict_init_net.RunAllOnGPU()
    init_net.RunAllOnGPU()

    assert workspace.RunNetOnce(predict_init_net)
    assert workspace.RunNetOnce(init_net)


""" Display Utils """

def GetTimeRemain(i, total_i, start):
    passed = time.time() - start
    remain_time = passed * (total_i - i) / i

    return '%d:%02d' % (int(remain_time / 60), remain_time % 60)

""" Defination of Train Model """

train_model = cnn.CNNModelHelper(
    order="NCHW",
    name=MODEL_NAME + "_train",
    init_params=not LOAD_MODEL
)
data, label = AddInput(
    train_model, batch_size=10,
    db=os.path.join(DATASET_FOLDER, DATASET_NAME + '_train.minidb'),
    db_type='minidb'
)
softmax = AddNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)


""" Defination of Test Model """

test_model = cnn.CNNModelHelper(
    order="NCHW",
    name=MODEL_NAME + "_test",
    init_params=False
)
data, label = AddInput(
    test_model, batch_size=1000,
    db=os.path.join(DATASET_FOLDER, DATASET_NAME + '_test.minidb'),
    db_type='minidb'
)
softmax = AddNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)


""" Net Initialization """

workspace.ResetWorkspace()

if USE_GPU:
    train_model.param_init_net.RunAllOnGPU()
    train_model.net.RunAllOnGPU()
    test_model.param_init_net.RunAllOnGPU()
    test_model.net.RunAllOnGPU()

if LOAD_MODEL:
    LoadModel(PRETRAIN_MODEL_PATH)
    
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)


""" Training Initialization"""

start_time = time.time()
loss = []
train_accuracys = [[], [], [], []]
val_accuracy = []


""" Training """

for i in range(1, TOTAL_ITERS + 1):
    # Training Phase
    workspace.RunNet(train_model.net.Proto().name)
   
    train_a1 = workspace.FetchBlob('accuracy0')
    train_a2 = workspace.FetchBlob('accuracy1')
    train_a3 = workspace.FetchBlob('accuracy2')
    train_a4 = workspace.FetchBlob('accuracy3')
    
    train_loss = workspace.FetchBlob('loss')

    if i % 50 == 0:
        os.system('clear')
        train_accuracys[0].append(train_a1)
        train_accuracys[1].append(train_a2)
        train_accuracys[2].append(train_a3)
        train_accuracys[3].append(train_a4)
        loss.append(train_loss)
        
        print ' TimeRemain <', GetTimeRemain(i, TOTAL_ITERS, start_time), '>'
        print ' ----> [' + ('=' * int(i * TB_LEN / TOTAL_ITERS) + '>').ljust(TB_LEN) + '] <----'
        print ' - Iter -', i, '/', TOTAL_ITERS
        print ' - Loss - ', train_loss
        print ' - Label1AC: ', train_a1
        print ' - Label2AC: ', train_a2
        print ' - Label3AC: ', train_a3
        print ' - Label4AC: ', train_a4
        
    
    # Val Phase
    if i % 200 == 0:
        workspace.RunNet(test_model.net.Proto().name)

        a1 = ([s.argmax() for s in workspace.FetchBlob('softmax0')] - workspace.FetchBlob('label1'))
        a2 = ([s.argmax() for s in workspace.FetchBlob('softmax1')] - workspace.FetchBlob('label2'))
        a3 = ([s.argmax() for s in workspace.FetchBlob('softmax2')] - workspace.FetchBlob('label3'))
        a4 = ([s.argmax() for s in workspace.FetchBlob('softmax3')] - workspace.FetchBlob('label4'))
        correct_num = 1000 - np.count_nonzero(a1 | a2 | a3 | a4)
    
        val_accuracy.append(correct_num / 1000.0)
        print ' - Test AC - %f' % (correct_num / 1000.0)

        if SAVE_MODEL:
            SaveModel(train_model, MODEL_SAVE_FOLDER, i)


""" Testing """

test_accuracy = []
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)

    a1 = ([s.argmax() for s in workspace.FetchBlob('softmax0')] - workspace.FetchBlob('label1'))
    a2 = ([s.argmax() for s in workspace.FetchBlob('softmax1')] - workspace.FetchBlob('label2'))
    a3 = ([s.argmax() for s in workspace.FetchBlob('softmax2')] - workspace.FetchBlob('label3'))
    a4 = ([s.argmax() for s in workspace.FetchBlob('softmax3')] - workspace.FetchBlob('label4'))
    correct_num = 1000 - np.count_nonzero(a1 | a2 | a3 | a4)
   
    acc_str = 'Test %d: ' % i
    for j, item in enumerate([a1, a2, a3, a4]):
        acc_str = acc_str + 'label%d-acc: %f, ' % (j, (1000 - np.count_nonzero(item)) / 1000.0)
    print acc_str

    test_accuracy.append(correct_num / 1000.0)

print('test_accuracy: %f' % np.mean(test_accuracy))
