import torch
from torch import nn

### Functions for Replacing Layer Norm with Power Norm
### Source: https://github.com/sIncerass/powernorm/blob/master/fairseq/modules/norms/mask_powernorm.py
### Distributed under MIT License.
class PowerFunction(torch.autograd.Function):
    @staticmethod

    def forward(ctx, x, weight, bias, running_phi, eps, afwd, abkw, ema_gz, \
                debug, warmup_iters, current_iter, mask_x, num_of_batch, accumulated_var, accumulation_step):
        ctx.eps = eps
        ctx.debug = debug
        current_iter = current_iter.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters
        ctx.abkw = abkw
        N, C, H, W = x.size()

        #mask_x = (128*32, 768)
        x2 = (mask_x * mask_x).sum(dim=0)/num_of_batch
        
        var = x2.reshape(1, C, 1, 1)
        if current_iter <= warmup_iters:
            z = x /(var + eps).sqrt()
        else:
            z = x /(running_phi + eps).sqrt()
            
        y = z
        ctx.save_for_backward(z, var, weight, ema_gz)
        # For accumulation

        accumulated_var.copy_(accumulated_var + var/accumulation_step)
        if (current_iter % accumulation_step == 0):
            if (current_iter < warmup_iters):
                running_phi.copy_(running_phi * (current_iter-1)/current_iter + accumulated_var.mean(dim=0, keepdim=True)/current_iter)
            running_phi.copy_(afwd*running_phi + (1-afwd)*accumulated_var.mean(dim=0, keepdim=True))
            accumulated_var.copy_(torch.zeros(1,C,1,1))

        
        y = weight.reshape(1,C,1,1) * y + bias.reshape(1,C,1,1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        abkw = ctx.abkw

        N, C, H, W = grad_output.size()
        z, var, weight, ema_gz = ctx.saved_variables

        y = z
        g = grad_output * weight.reshape(1, C, 1, 1)
        g = g * 1
        
        approx_grad_g = (g - (1 - abkw) * ema_gz * z)
        ema_gz.add_((approx_grad_g * z).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True))

        gx = 1. / torch.sqrt(var + eps) * approx_grad_g 
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), \
         None, None, None, None, None, None, None, None, None, None, None, None

### Source: https://github.com/sIncerass/powernorm/blob/master/fairseq/modules/norms/mask_powernorm.py
### Distributed under MIT License.
class MaskPowerNorm(nn.Module):
    """
    An implementation of masked power normalization, used for testing the numerical
    stability.
    """

    def __init__(self, num_features, eps=1e-5, alpha_fwd=0.9, alpha_bkw=0.9, \
                affine=True, warmup_iters=10000, accumulation_step=8):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        self.register_parameter('weight', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_features)))
        self.register_buffer('running_phi', torch.ones(1,num_features,1,1))
        self.register_buffer('ema_gz', torch.zeros(1,num_features,1,1))
        self.register_buffer('iters', torch.zeros(1).type(torch.LongTensor))
        self.register_buffer('accumulated_var', torch.zeros(1,num_features,1,1))

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        self.eps = eps
        self.debug = False
        self.warmup_iters = warmup_iters
        self.accumulation_step = accumulation_step

    def extra_repr(self):
        return '{num_features}, eps={eps}, alpha_fwd={afwd}, alpha_bkw={abkw}, ' \
               'affine={affine}, warmup={warmup_iters}'.format(**self.__dict__)

    def forward(self, input, input_mask=None):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        shaped_input = (len(input.shape) == 2)
        if shaped_input:
            input = input.unsqueeze(0)
        T, B, C = input.shape

        # construct the mask_input, size to be (BxL) x C: L is the real length here
        if input_mask is None:
            input_mask = torch.ones((T*B, 1))
            masked_input = input.contiguous().view(-1, C)
        else:
            # Transpose the bn_mask (B x T -> T x B)
            masked_input = (input * input_mask).squeeze(0)

        input = input.permute(1, 2, 0).contiguous()
        input_shape = input.size()
        input = input.reshape(input.size(0), self.num_features, -1)
        input = input.unsqueeze(-1)
        num_of_batch = input_mask.sum()
        if self.training:
            self.iters.copy_(self.iters + 1)
            output = PowerFunction.apply(input, self.weight, self.bias, self.running_phi, self.eps, \
                        self.afwd, self.abkw, self.ema_gz, self.debug, self.warmup_iters, self.iters, masked_input, num_of_batch,
                        self.accumulated_var, self.accumulation_step
                        )
            
        else:
            N, C, H, W = input.size()
            var = self.running_phi
            output = input / (var + self.eps).sqrt()
            output = self.weight.reshape(1,C,1,1) * output + self.bias.reshape(1,C,1,1)

        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        # Reshape it.
        if shaped_input:
            output = output.squeeze(0)

        return output

def NormSelect(config):
    if config.norm_type == "Layer":
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    elif config.norm_type == 'Power':
        return MaskPowerNorm(config.hidden_size, warmup_iters=config.warmup_updates, accumulation_step=config.accumulation_step, eps=config.layer_norm_eps)

# Function for Replacing Softmax with Neural Network Approximation
class Softmax_Approximator(nn.Module):
    def __init__(self, input_size = 128, hidden_size = 128):
        super(Softmax_Approximator, self).__init__()
        #Neural Network with 1 hidden layer (ReLU) and 1 output layer
        self.linear_relu_stack = nn.Sequential(
            #Intermediate Layer with ReLU
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            #Output Layer
            nn.Linear(hidden_size, input_size))
        
    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output
