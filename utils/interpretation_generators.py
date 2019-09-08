import torch
import torch.distributions as tdist

def simple_gradient(net, sample, label, used=True):
    """Creates simple gradient saliency map.
    """
    if not used:
        sample = torch.autograd.Variable(sample, requires_grad=True)
    net(sample)
    logits = net.logits
    bp_mat = torch.nn.functional.one_hot(label, num_classes=10).float()
    grads = torch.autograd.grad(logits * bp_mat, sample, grad_outputs=bp_mat, create_graph=True)[0].squeeze()
    # equivalent to
#     logit = torch.sum(bp_mat * logits)
#     grads = torch.autograd.grad(logit, sample)[0].squeeze()
    saliency = torch.abs(grads)
    saliency = saliency/torch.sum(saliency)
    return saliency

def smoothgrad(net, sample, label, j=15, scale=1., used=True):
    """Creates smoothgrad saliency map.
    """
    if not used:
        sample = torch.autograd.Variable(sample, requires_grad=True)
    normal = tdist.Normal(loc=torch.tensor([0.]), scale=torch.tensor([scale]))
    shape = list(sample.shape)
    shape[0] = j
    samples = sample + normal.sample(shape).reshape(shape)
    samples = torch.cat([samples, sample], dim=0)
    net(samples)
    logits = net.logits 
    bp_mat = torch.nn.functional.one_hot(label.repeat(j+1,1), num_classes=10).float().squeeze()
    grads = torch.autograd.grad(logits * bp_mat, samples, grad_outputs=bp_mat, create_graph=True)[0].squeeze()
    saliency = torch.abs(grads)
    saliency = saliency.mean(dim=0)
    saliency = saliency/torch.sum(saliency)
    return saliency

def simple_gradients(net, samples, labels, used=True):
    """Creates simple gradient saliency map.
    """
    if not used:
        samples = torch.autograd.Variable(samples, requires_grad=True)
    net(samples)
    logits = net.logits
    bp_mat = torch.nn.functional.one_hot(labels, num_classes=10).float()
    grads = torch.autograd.grad(logits * bp_mat, samples, grad_outputs=bp_mat, create_graph=True)[0].squeeze()
    # equivalent to
#     logit = torch.sum(bp_mat * logits)
#     grads = torch.autograd.grad(logit, sample)[0].squeeze()
    saliency = torch.abs(grads)
#     saliency = saliency/torch.sum(saliency)
    return saliency