import sys
import os
import numpy as np

import mxnet as mx
from ..model_zoo.vgg import VGG19AvgOutput, VGG19Avg, get_vgg19_avg
from .. import PreprocessImage, PostprocessImage
from . import __get_tv_grad, __get_style_gram

from tqdm import tqdm

class ClassicTransferArgs:
    __slots__ = ['content_weight', 'tv_weight', 'style_weight', 'lr_sched_delay',
            'lr_sched_factor', 'stop_eps', 'content_feature',
            'style_feature', 'learning_rate', 'epochs', 'color_preserve']

    def __init__(self):
        self.epochs = 1000
        self.tv_weight = 0.02
        self.content_weight = 10
        self.style_weight = 1.0
        self.lr_sched_delay = 50
        self.lr_sched_factor = 0.5
        self.stop_eps = 0.0001
        self.color_preserve = False
        self.content_feature = [VGG19AvgOutput.relu4_2]
        self.style_feature = [VGG19AvgOutput.relu1_1, VGG19AvgOutput.relu2_1,
                VGG19AvgOutput.relu3_1, VGG19AvgOutput.relu4_1, VGG19AvgOutput.relu5_1]
        self.learning_rate = 0.001

def classic_transfer(content_img, style_img, args, ctx = None, verbose = True):
    """Do style transfer with content and style image, this is an implementation
    of paper https://arxiv.org/abs/1508.06576.

    Parameter
    _________

    content_img : mx.ndarray.NDArray
            This points to ndarray of image. The shape of which could be (3, w, h)
            or (w, h, 3), and it does not need to subtract mean of ImageNet, which
            will be processed internally.

    style_img : mx.ndarray.NDArray
            This points to ndarray of image. The shape requirement is the same as content_img.

    args : ClassicTransferArgs
            Training hyperparameters.

    ctx : mx.cpu or mx.gpu

    Return
    ______

    This function is iterable that each epoch will yeild the transfered image currently generated.
    """

    assert isinstance(args, ClassicTransferArgs), 'Args should be instance of ClassicTransferArgs'

    content_img = PreprocessImage(content_img).copyto(ctx)
    style_img = PreprocessImage(style_img).copyto(ctx)

    # load pretrained vgg19
    vgg19avg = get_vgg19_avg(pretrained = True)
    # style = [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
    # content = [relu4_2]
    input = mx.sym.var('data')
    style_content_symbols = vgg19avg.get_output_symbols(input, args.style_feature, args.content_feature),

    style_content_net = mx.gluon.SymbolBlock(inputs = input, outputs = style_content_symbols, params = vgg19avg.collect_params())
    style_content_net.collect_params().reset_ctx(ctx)

    # extract target content and style
    target = style_content_net(content_img)[0]
    content_targets = target[len(args.style_feature):]
    target = style_content_net(style_img)[0]
    style_targets = target[:len(args.style_feature)]

    # compute target gram matrix
    target_gram_list, gram_scale_list = __get_style_gram(style_targets)

    # Generate random image to do style transfer
    random_img = mx.nd.random_uniform(-0.1, 0.1, content_img.shape, ctx = ctx)
    clip_norm = np.prod(random_img.shape)

    # optimizer
    lr = mx.lr_scheduler.FactorScheduler(step = args.lr_sched_delay, factor = args.lr_sched_factor)
    optimizer = mx.optimizer.NAG(learning_rate = args.learning_rate, wd = 0.0001,
            momentum = 0.95, lr_scheduler = lr)

    # This is needed for momentum
    optim_state = optimizer.create_state(0, random_img)

    # Training and transfer
    random_img.attach_grad() # attach grad for update
    for epoch in tqdm(range(args.epochs)):
        with mx.autograd.record():
            style_content = style_content_net(random_img)[0]
            contents = style_content[len(args.style_feature):]
            styles = style_content[:len(args.style_feature)]

            gram_list, _ = __get_style_gram(styles)
            total_loss = 0
            for content, target_content in zip(contents, content_targets):
                loss = mx.nd.sum(mx.nd.square(content - target_content))
                total_loss = total_loss + loss * args.content_weight

            for gram, target_gram, gscale in zip(gram_list, target_gram_list, gram_scale_list):
                loss = mx.nd.sum(mx.nd.square(gram - target_gram))
                total_loss = total_loss + loss * args.style_weight / gscale

        total_loss.backward()

        gnorm = mx.nd.norm(random_img.grad).asscalar()
        if gnorm > clip_norm:
            random_img.grad[:] *= clip_norm / gnorm

        if verbose:
            print('Training: epoch %d, loss: %f' % (epoch, total_loss.asscalar()))

        old_img = random_img.copy()
        tv_grad = __get_tv_grad(random_img, ctx, args.tv_weight)
        optimizer.update(0, random_img, random_img.grad + tv_grad, optim_state)

        eps = (mx.nd.norm(old_img - random_img) / mx.nd.norm(random_img)).asscalar()
        if eps < args.stop_eps:
            print('eps (%f) < args.stop_eps (%f), training finished' % (eps, args.stop_eps))
            break

        yield PostprocessImage(random_img)
    yield PostprocessImage(random_img)
