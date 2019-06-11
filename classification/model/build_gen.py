from model import svhn2mnist
from model import usps_model
from model import syn2gtrsb
from model import visda_model
## where is that?
# import syndig2svhn

def Generator(source, target, pixelda=False, resnet_arg='101'):
    if source == 'usps' or target == 'usps':
        return usps_model.Feature()
    elif source == 'svhn' or target == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()
    elif source == 'visda':
        option = 'resnet' + resnet_arg
        return visda_model.ResBase(option)



def Classifier(source, target, num_layer=2):
    if source == 'usps' or target == 'usps':
        return usps_model.Predictor()
    if source == 'svhn' or target == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()
    if source == 'visda':
        Predictor = visda_model.ResClassifier(num_layer=num_layer)
        Predictor.apply(visda_model.weights_init)
        return Predictor

