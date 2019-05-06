import svhn2mnist
import usps_model
import syn2gtrsb
## where is that?
# import syndig2svhn

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps_model.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps_model.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()

