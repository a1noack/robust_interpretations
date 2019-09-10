import torch
import torch.distributions as tdist

def smoothgrad(net, sample, label, j=15, scale=1., used=True):
    """Creates smoothgrad saliency map. Unparallelized.
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

def simple_gradient(net, samples, labels, used=True):
    """Parallelized version of simple gradient saliency map function.
    """
    if not used:
        samples = torch.autograd.Variable(samples, requires_grad=True)
    net(samples)
    logits = net.logits
    bp_mat = torch.nn.functional.one_hot(labels, num_classes=10).float()
    grads = torch.autograd.grad(logits * bp_mat, samples, grad_outputs=bp_mat, create_graph=True)[0].squeeze()
    saliency = torch.abs(grads)
    # normalize each saliency map by dividing each component of each saliency map
    # by the sum of the saliency map
    totals = saliency.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).repeat(1,saliency.shape[-2],saliency.shape[-1])
    saliency = torch.div(saliency, totals)
    return saliency