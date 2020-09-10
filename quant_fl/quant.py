import torch
import torch.nn as nn

def quantizer(input, nbit):
    '''
    input: full precision tensor in the range [0, 1]
    return: quantized tensor
    '''
    if nbit<32:
        sf = 2**nbit - 1.0
        return torch.round(sf * input) / sf
    else:
        return input

# def quantizer2(input, nbit):
#     '''
#     input: full precision tensor in the range [0, 1]
#     return: quantized tensor
#     '''
#     # Part 3.1: Implement!
#     input *= ((2 ** nbit) - 1)  # scale input by 2^n - 1
#     input = input.round()  # round the scaled input
#     input *= 1 / float(((2 ** nbit) - 1))  # scale the input back down
#     return input

def dorefa_g(w, nbit, adaptive_scale=None):
    '''
    w: a floating-point weight tensor to quantize
    nbit: the number of bits in the quantized representation
    adaptive_scale: the maximum scale value. if None, it is set to be the
                    absolute maximum value in w.
    '''
    if adaptive_scale is None:
        adaptive_scale = torch.max(torch.abs(w))

    # Part 3.2: Implement based on stochastic quantization function above
    w = w / (2 * adaptive_scale) + 0.5
    w += torch.empty_like(w).uniform_(-0.5, 0.5)/(2**nbit-1.0)
    # print(w[0])
    w = 2 * adaptive_scale * (quantizer(w, nbit) - 0.5)

    return w, adaptive_scale

# def dorefa_g(w, nbit, adaptive_scale=None):
#     '''
#     w: a floating-point weight tensor to quantize
#     nbit: the number of bits in the quantized representation
#     adaptive_scale: the maximum scale value. if None, it is set to be the
#                     absolute maximum value in w.
#     '''
#     if adaptive_scale is None:
#         adaptive_scale = torch.max(torch.abs(w))

#     # Part 3.2: Implement based on stochastic quantization function above

#     w_tilda = (w / (2 * float(adaptive_scale))) + 0.5  # w_tilda defined above
#     sig = torch.distributions.uniform.Uniform(-0.5, 0.5).sample(sample_shape=w_tilda.shape).cuda()
#     sig = torch.empty_like(w).uniform_(-0.5, 0.5)
#     w_tilda += sig / float((2 ** nbit) - 1)  # add stochastic part

#     w_hat = quantizer(w_tilda, nbit)  # returns n-bit quantized w_tilda

#     w_q = (2 * adaptive_scale) * (w_hat - 0.5)  # subtract 1/2 and multiply by 2*adaptive_scale (as per the spec)

#     return w_q, adaptive_scale




def quantize_model(model, nbit):
    '''
    Used in Code Cell 3.3 to quantize the ConvNet model
    '''
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data, m.adaptive_scale = dorefa_g(m.weight, nbit)
            if m.bias is not None:
                m.bias.data,_ = dorefa_g(m.bias, nbit, m.adaptive_scale)


def sign_state_dict(state_dict):
    for n, t in state_dict.items():
        t.sign_()
    return state_dict