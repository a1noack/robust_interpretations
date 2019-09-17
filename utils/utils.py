import torch
import torch.nn.functional as F
import utils.adv_ex_utils as aus
import utils.interp_generators as igs
from torch.autograd import Variable

def bp_matrix(batch_size, n_outputs):
    """Creates matrix that is used to calculate Jacobian for multiple input 
    samples at once.
    """
    idx = torch.arange(n_outputs).reshape(n_outputs,1).repeat(1,batch_size).reshape(batch_size*n_outputs,)
    
    return torch.zeros(len(idx), n_outputs).scatter_(1, idx.unsqueeze(1), 1.)

def rescale(sample):
    """Rescales RGB image so that each pixel value is between zero and one.
    """
    sample = (sample+sample.min())/(sample.max()-sample.min())
    return sample

def avg_norm_jacobian(net, x, n_outputs, bp_mat, for_loss=True):
    """Returns squared frobenius norm of the input-output Jacobian averaged 
    over the entire batch of inputs in x.
    """
    batch_size = x.shape[0]
    # needed because some edge-case batches are not standard size
    if bp_mat.shape[0]/n_outputs != batch_size:     
        bp_mat = bp_matrix(batch_size, n_outputs)
    x = x.repeat(n_outputs, 1, 1, 1)
    x = Variable(x, requires_grad=True)
    # needed so that we can get gradient of output w.r.t input
    y = net(x)
    x_grad = torch.autograd.grad(y, x, grad_outputs=bp_mat, create_graph=for_loss)[0]
    # get sum of squared values of the gradient values
    j = x_grad.pow(2).sum() / batch_size
    
    return j

def norm_diff_interp(net, x, labels, dataset, ig=igs.simple_gradient, scale=.1):   
    """Gets the norm of the difference between the interpretations generated
    at original data points x and randomly perturbed points x_.
    Also returns the interpretations generated for all of the data points.
    """
    x_ = aus.perturb_randomly(x, dataset, scale=scale)
    ix = ig(net, x, labels, normalize=False, used=False)
    ix_ = ig(net, x_, labels, normalize=False, used=False)
    if torch.isnan(ix).any() or torch.isnan(ix_).any():
        print(f'{torch.sum(torch.isnan(ix))/ix.shape[0]:2f} of ix are nan')

    return torch.norm(torch.abs(ix-ix_)), torch.cat([ix, ix_], dim=0)

def my_loss(output, labels, net=None, 
            optimizer=None, alpha_wd=0, alpha_jr=0, 
            x=None, bp_mat=None, alpha_ir1=0, alpha_ir2=0, scale=.1, dataset='MNIST'):
    """Adds terms for L2-regularization and the norm of the input-output 
    Jacobian to the standard cross-entropy loss function. Check https://arxiv.org/abs/1908.02729
    for alpha_wd, alpha_jr suggestions.
    """
    # standard cross-entropy loss base
    loss = F.cross_entropy(output, labels)
    
    # add l2 regularization to loss 
    if alpha_wd != 0:
        l2 = 0
        for p in net.parameters():
            l2 += p.pow(2).sum()
        loss = loss + alpha_wd * l2
    
    # add input-output jacobian regularization formulation
    if alpha_jr != 0:
        n_outputs = output.shape[1]
        j = avg_norm_jacobian(net, x, n_outputs, bp_mat)
        loss = loss + (alpha_jr / 2) * j
        # needed so gradients don't accumulate in leaf variables when calling loss.backward in train function
        optimizer.zero_grad()
    
    # add one or both of the interpretation regularization terms
    if alpha_ir1 != 0 or alpha_ir2 != 0:
        norm, ix = norm_diff_interp(net, x, labels, dataset=dataset, scale=scale)
        
        # add norm of difference between interpretations generated at x and x_
        if alpha_ir1 != 0:
            loss = loss + alpha_ir1 * norm
            optimizer.zero_grad()
        
        # add l0 interpretation regularization
        if alpha_ir2 != 0:
            # differentiable approximation of l0 norm:
            l0_approx = torch.sum(torch.abs(ix / (torch.abs(ix) + .0001)))
            loss = loss + alpha_ir2 * l0_approx
            optimizer.zero_grad()

    return loss