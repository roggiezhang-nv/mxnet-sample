import sys
import os
from enum import Enum

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential

class VGG19AvgOutput(Enum):
    conv1_1 = 'conv1_1'
    relu1_1 = 'relu1_1'
    conv1_2 = 'conv1_2'
    relu1_2 = 'relu1_2'
    conv2_1 = 'conv2_1'
    relu2_1 = 'relu2_1'
    conv2_2 = 'conv2_2'
    relu2_2 = 'relu2_2'
    conv3_1 = 'conv3_1'
    relu3_1 = 'relu3_1'
    conv3_2 = 'conv3_2'
    relu3_2 = 'relu3_2'
    conv3_3 = 'conv3_3'
    relu3_3 = 'relu3_3'
    conv3_4 = 'conv3_4'
    relu3_4 = 'relu3_4'
    conv4_1 = 'conv4_1'
    relu4_1 = 'relu4_1'
    conv4_2 = 'conv4_2'
    relu4_2 = 'relu4_2'
    conv4_3 = 'conv4_3'
    relu4_3 = 'relu4_3'
    conv4_4 = 'conv4_4'
    relu4_4 = 'relu4_4'
    conv5_1 = 'conv5_1'
    relu5_1 = 'relu5_1'

class VGG19AvgPretrainedInitializer(mx.init.Initializer):

    def __init__(self, prefix, params, verbose = False):
        self.prefix_len = len(prefix)
        self.verbose = verbose
        self.arg_params = {k : v for k, v in params.items() if k.startswith('arg:')}
        self.aux_params = {k : v for k, v in params.items() if k.startswith('aux:')}
        self.arg_names = set([k[4:] for k in self.arg_params.keys()])
        self.aux_names = set([k[4:] for k in self.aux_params.keys()])


    def __call__(self, name, arr):
        key = name[self.prefix_len:]
        if key in self.arg_names:
            if self.verbose:
                print("Init %s" % name)
            self.arg_params['arg:' + name].copyto(arr)
        elif key in self.aux_names:
            if self.verbose:
                print("Init %s" % name)
            self.aux_params['aux:' + name].copyto(arr)
        else:
            print("Unknow params: %s, init with 0" % name)
            arr[:] = 0

class VGG19Avg(HybridBlock):

    def __init__(self, prefix = '', **kwargs):
        super(VGG19Avg, self).__init__(prefix = prefix, **kwargs)

        with self.name_scope():
            self.features = HybridSequential()
            self.__conv_block(1, 2, 64)
            self.__conv_block(2, 2, 128)
            self.__conv_block(3, 4, 256)
            self.__conv_block(4, 4, 512)
            self.__conv_block(5, 1, 512, False)

    def __conv_block(self, layer, num_conv, filters, include_pool = True):
        for l in range(num_conv):
            self.features.add(gluon.nn.Conv2D(filters, 3, strides = 1, padding = 1, prefix = ('conv%d_%d_' % (layer, l + 1))))
            self.features.add(gluon.nn.Activation('relu', prefix = ('relu%d_%d_' % (layer, l + 1))))
        if include_pool:
            self.features.add(gluon.nn.AvgPool2D(strides = 2, prefix = ('pool%d_' % layer)))


    def hybrid_forward(self, F, x):
        x = self.features(x)

        return x

    def get_output_symbols(self, input, *args):
        assert isinstance(input, mx.symbol.Symbol), "Input must be mx.symbol.Symbol"
        assert len(args) > 0, "Args should not be empty."

        names = []
        for arg in args:
            if isinstance(arg, list):
                names.extend(arg)
            else:
                names.extend([arg])

        output = self.forward(input)
        internals = output.get_internals()

        symbols = []
        for name in names:
            assert isinstance(name, VGG19AvgOutput), 'Parameter of name must be scalar or list of VGG19AvgOutput'
            symbols.extend(internals[name.value + '_fwd_output'])

        return symbols


def get_vgg19_avg(pretrained = False, ctx = None, prefix = '', verbose = False):
    net = VGG19Avg(prefix)
    if pretrained:
        MYDIR = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(MYDIR, '..', '..', '..', '..', 'pretrained_models', 'vgg19_avg_no_fc.params')
        assert os.path.exists(params_path), 'Cannot find pretrained model params under %s' % params_path

        params = mx.nd.load(params_path)

        net.collect_params().initialize(VGG19AvgPretrainedInitializer(prefix = prefix,
            params = params, verbose = verbose), ctx = ctx, verbose = verbose)

    return net

if __name__ == '__main__':
    vgg19 = get_vgg19_avg(pretrained = True, ctx = mx.cpu(), prefix = '', verbose = False)

    input = mx.sym.var('data')
    symbols = vgg19.get_output_symbols(data, input, VGG19AvgOutput.relu5_4)
    print(symbols[0].list_inputs())
