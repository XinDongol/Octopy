from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import os

import random
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter

import quant


def image_preprocess(tensor):
    # sigmoid
    tensor = F.sigmoid(tensor)
    # quant
    tensor = quant.pixel_quant(tensor, nbit=8, in_place=False)
    # normalize
    tensor = quant.ImageNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)).normalize(tensor, False)
    
    return tensor


class DeepInversionClass(object):
    def __init__(self, bs=10,
                 net_teacher=None,
                 path="/raid/gen_images/"):

        # for reproducibility
        torch.manual_seed(torch.cuda.current_device())

        self.net_teacher = net_teacher

        self.image_resolution = 32
        self.do_flip = True
        self.store_best_images = False

        self.bs = bs  # batch size
        self.save_every = -1
        self.jitter = 2

        self.bn_reg_scale = 10.
        self.first_bn_multiplier = 1.
        self.var_scale_l1 = 0.0
        self.var_scale_l2 = 2.5e-5
        self.l2_scale = 0.0
        self.lr = 0.05
        self.main_loss_multiplier = 1.0

        # Create folders for images and logs
        create_folder(path + "/best_images/")

        # Create hooks for feature statistics
        self.loss_r_feature_layers = []
        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(
                    DeepInversionFeatureHook(module))

        # init tensorboard writer
        self.writer = SummaryWriter(path+'/log/')

    def get_images(self, net_student, reset_inputs, reset_targets, reset_opt, iterations_per_layer=2000, quant_input=False):
        writer = self.writer
            
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        best_cost = 1e4
        criterion = nn.CrossEntropyLoss()

        net_teacher = self.net_teacher
        net_teacher.eval()
        if net_student:
            net_student.eval()

        # init setting
        save_every = self.save_every

        
        
        do_clip = True

        self.all_teacher_output = []
        self.all_verifier_output = []

        # setup target labels and inputs
        if (not hasattr(self, 'targets')) or (reset_targets):
            # only works for classification now, for other tasks need to provide target vector
            self.targets = torch.LongTensor(
                [random.randint(0, 10) for _ in range(self.bs)]).to('cuda')
            
        if (not hasattr(self, 'inputs')) or (reset_inputs):

            if quant_input:
                self.inputs = torch.tensor(np.random.uniform(low=-5.0,high=5.0,size=[self.bs, 3, self.image_resolution, self.image_resolution]),
                                           requires_grad=True, device=torch.device('cuda'), dtype=torch.float)
            else:
                self.inputs = random_images(bsz=self.bs, nch=3, h=self.image_resolution, w=self.image_resolution,
                           means=(0.4914, 0.4822, 0.4465), stds=(0.2023, 0.1994, 0.2010),
                           use_cuda=True)


        if (not hasattr(self, 'optimizer')) or (reset_inputs):
            # self.optimizer = optim.Adam([self.inputs, self.raw_targets], lr=self.lr,
            #                    betas=[0.5, 0.9], eps=1e-8)
            self.optimizer = optim.Adam([self.inputs], lr=self.lr)
            # iterations_per_layer = 2000
        else:
            i# terations_per_layer = 10
        # lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

        
        iteration = 0
        for iteration_loc in range(iterations_per_layer):
            iteration += 1
            # call the learning rate scheduling
            # lr_scheduler(optimizer, iteration_loc, iteration_loc)
            # args for `lr_scheduler`: optimizzer, iteration, epoch

            if quant_input:
                inputs_jit = image_preprocess(self.inputs)
            else:
                inputs_jit = self.inputs
            
            

            
            # apply random jitter offsets
            off1 = random.randint(-self.jitter, self.jitter)
            off2 = random.randint(-self.jitter, self.jitter)
            # https://pytorch.org/docs/stable/torch.html?highlight=roll#torch.roll
            inputs_jit = torch.roll(
                inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # Flipping
            flip = random.random() > 0.5
            if flip and self.do_flip:
                inputs_jit = torch.flip(inputs_jit, dims=(3,))
            

            # forward pass
            self.optimizer.zero_grad()
            net_teacher.zero_grad()

            outputs = net_teacher(inputs_jit)

            

            #! R_cross classification loss
            # loss = criterion(outputs, self.targets)
            loss = 0
            # self.targets = F.softmax(self.raw_targets, dim=1)
            # loss = kl_loss(F.log_softmax(outputs, dim=1), self.targets)
            # loss_cross = loss.item()  # record this loss
            loss_cross = 0 

            #! R_prior losses
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

            #! l2 loss on images
            loss_l2 = torch.norm(inputs_jit, 2)

            #! R_feature loss
            rescale = [self.first_bn_multiplier] + \
                [1. for _ in range(len(self.loss_r_feature_layers)-1)]
            loss_r_feature = sum([mod.r_feature * rescale[idx]
                                  for (idx, mod) in enumerate(self.loss_r_feature_layers)])

        
            #! combining losses
            loss_aux = self.var_scale_l2 * loss_var_l2 + \
                self.var_scale_l1 * loss_var_l1 + \
                self.bn_reg_scale * loss_r_feature + \
                self.l2_scale * loss_l2


            loss = self.main_loss_multiplier * loss + loss_aux

            # do image update
            loss.backward()
            self.optimizer.step()

            # clip color outlayers
            if do_clip:
                self.inputs.data = clip(self.inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = self.inputs.data.clone()

            if iteration % 2000 == 0:
                print('-------------------- %d ------------------' % iteration)
                print('R_feature: %.3e, R_cross: %.3e, R_total: %.3e'
                      % (loss_r_feature.item(), loss_cross, loss.item()))

            writer.add_scalar('Loss/prior_var_l1', loss_var_l1.item(), iteration)
            writer.add_scalar('Loss/prior_var_l2', loss_var_l2.item(), iteration)
            writer.add_scalar('Loss/image_l2', loss_l2.item(), iteration)
            writer.add_scalar('LR/lr', next(iter(self.optimizer.param_groups))['lr'], iteration)
            writer.add_scalar('Loss/r_feature', loss_r_feature.item(), iteration)
            writer.add_scalar('Loss/r_cross', loss_cross, iteration)

            # if iteration % save_every == 0 and (save_every > 0):
            #     if local_rank == 0:
            #         vutils.save_image(inputs,
            #                           '{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
            #                                                                            iteration // save_every,
            #                                                                            local_rank),
            #                           normalize=True, scale_each=True, nrow=int(10))

        # if self.store_best_images:
        #     best_inputs = denormalize(best_inputs)
        #     self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        self.optimizer.state = collections.defaultdict(dict)

        if quant_input:
            return image_preprocess(self.inputs), self.targets
        else:
            return self.inputs, self.targets
        
    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if 0:
                # save into separate folders
                place_to_store = '{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                      self.num_generations, id,
                                                                                      local_rank)
            else:
                place_to_store = '{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                      self.num_generations, id,
                                                                                      local_rank)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view(
            [nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        # print('running mean: ', module.running_mean.data.size())
        # print('real mean: ', mean.size())

        # r_feature = margin_l2(module.running_var.data, var, margin=module.running_var.data*0.05)\
        #     + margin_l2(module.running_mean.data, mean, margin=module.running_mean.data*0.05)

        # print('r_feature: ', r_feature)
        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def margin_l2(a, b, margin):
    return torch.sum(F.relu((a-b)**2-margin**2))**(1./2)


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + \
        torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
        diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def loss_fn_kd(outputs, teacher_outputs, labels, alpha=1.0, temperature=3.0, scaling_constant=1.):
    """
    Compute the knowledge-distillation (KD) loss given outputs

    :labels: can be None
    :scaling_constant: extra hyper-parameter to scale up the soft loss
    """
    T = temperature

    kld_loss = F.kl_div(F.log_softmax(outputs/T, dim=1),
                        F.softmax(teacher_outputs/T, dim=1),
                        reduction='batchmean')
    if labels is not None:
        hard_loss = F.cross_entropy(outputs, labels)
        final_loss = kld_loss * \
            (alpha*T*T*scaling_constant) + hard_loss*(1.-alpha)
    else:
        final_loss = kld_loss * (T*T) * scaling_constant

    return final_loss


def random_images(bsz, nch, h, w, means, stds, use_cuda=True):
    assert len(means) == len(stds) == nch

    images = torch.empty([bsz, nch, h, w], requires_grad=True, device='cuda' if use_cuda else 'cpu')

    for channel_idx in range(nch):
        torch.nn.init.normal_(images[:, channel_idx, :, :], mean=means[channel_idx], std=stds[channel_idx])

    return images


def clip(image_tensor):
    '''
    adjust the input based on mean and variance
    '''
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)
