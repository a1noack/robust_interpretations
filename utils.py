import torch
from advertorch.attacks import GradientSignAttack, CarliniWagnerL2Attack, PGDAttack

def generate_adversarial_samples(og_samples, true_labels, adversary, num_per_samp=1):
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