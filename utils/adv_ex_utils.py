import torch
import torch.distributions as tdist
from advertorch.attacks import GradientSignAttack, CarliniWagnerL2Attack, PGDAttack

def generate_adv_exs(og_samples, true_labels, adversary, num_per_samp=1):
    """Create num_per_samp adversarial examples for each sample in
    og_samples. Return the generated samples along with corresponding 
    adv_labels, a tensor containing the adversarial examples' labels.
    """
    adv_samples = []
    for i in range(num_per_samp):
        adv_samples.append(adversary.perturb(og_samples, true_labels))
    adv_samples = torch.cat(adv_samples, 0)
    adv_labels = torch.cat([true_labels]*num_per_samp, 0)
    
    return adv_samples, adv_labels

def perturb_randomly(og_samples, scale=.1):
    """Add a random variable drawn from a normal distribution with 
    specified parameters to each dimension of each sample and return.
    """
    normal = tdist.Normal(loc=torch.tensor([0.]), scale=torch.tensor([scale]))
    shape = list(og_samples.shape)
    pert_samples = og_samples + normal.sample(shape).reshape(shape)
    pert_samples = torch.clamp(pert_samples, min=-0.4242, max=2.8215)
    
    return pert_samples