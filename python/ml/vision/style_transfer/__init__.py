import mxnet as mx
import numpy as np

def __get_style_gram(styles):
    gram_list = []
    gram_scale_list = []
    for style in styles:
        shape = style.shape
        style = mx.nd.reshape(style, shape = (int(shape[1]), int(np.prod(shape[2:]))))
        gram = mx.nd.FullyConnected(style, style, no_bias = True, num_hidden = shape[1])
        gram = mx.nd.expand_dims(gram, axis = 0)
        gram_list.append(gram)
        gram_scale_list.append(shape[1] * np.prod(shape[1:]))

    return gram_list, gram_scale_list

def __get_tv_grad(img, ctx, tv_weight):
    if tv_weight <= 0:
        return 0

    kernel = mx.nd.array(np.array([[0, -1, 0,
                                -1, 4, -1,
                                0,  -1, 0]]).reshape(1, 1, 3, 3)
                                    , ctx = ctx) / 8.0
    nchannel = img.shape[1]
    channels = mx.nd.SliceChannel(img, num_outputs = nchannel)
    out = mx.nd.Concat(*[mx.nd.Convolution(data = channels[i],
        weight = kernel, num_filter = 1, kernel = (3, 3),
        stride = (1, 1), no_bias = True, pad = (1, 1)) for i in range(nchannel)])

    out = out * tv_weight

    return out



