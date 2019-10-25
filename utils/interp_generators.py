import torch
import torch.distributions as tdist

def smoothgrad(net, sample, label, normalize=True, abs_val=True, j=50, scale=1., used=True, for_loss=False):
    """Creates smoothgrad saliency map. Unparallelized.
    """
    if not used:
        sample = torch.autograd.Variable(sample, requires_grad=True)
    normal = tdist.Normal(loc=torch.tensor([0.]), scale=torch.tensor([scale]))
    shape = list(sample.shape)
    shape[0] = j
    samples = sample + normal.sample(shape).reshape(shape)
    samples = torch.cat([samples, sample], dim=0)
    logits = net(samples)
    bp_mat = torch.nn.functional.one_hot(label.repeat(j+1,1), num_classes=logits.shape[1]).float().squeeze()
    grads = torch.autograd.grad(logits, samples, grad_outputs=bp_mat, create_graph=for_loss)[0].squeeze()
    if abs_val:
        saliency = torch.abs(grads)
    saliency = saliency.mean(dim=0)
    if saliency.shape[0] == 3:
        saliency = torch.max(saliency, dim=0, keepdims=True)[0]
    if normalize:
        saliency = saliency/torch.sum(saliency)
        
    return saliency

def simple_gradient(net, samples, labels, normalize=True, abs_val=True, used=True, for_loss=False):
    """Parallelized version of simple gradient saliency map function.
    """
    if not used:
        samples = torch.autograd.Variable(samples, requires_grad=True)
    logits = net(samples)
    bp_mask = torch.nn.functional.one_hot(labels, num_classes=10).float()
#     label_logits = torch.sum(logits * bp_mask, dim=1, keepdim=True)
    grads = torch.autograd.grad(logits, samples, grad_outputs=bp_mask, create_graph=for_loss)[0]
    if abs_val:
        saliency = torch.abs(grads)
    # this is needed to compress 3 channel map to 1 channel
    # changed from mean to max
    if saliency.shape[1] == 3:
        saliency = torch.max(saliency, dim=1, keepdims=True)[0]
    # expected to have no channel dimension
    saliency = saliency.squeeze()
    # normalize each saliency map by dividing each component of each saliency map
    # by the sum of the saliency map
    if normalize:
        totals = saliency.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).repeat(1,saliency.shape[-2],saliency.shape[-1])
        saliency = torch.div(saliency, totals).squeeze()
    
    return saliency