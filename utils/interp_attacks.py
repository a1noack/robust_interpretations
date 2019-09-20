import torch
import scipy.stats
import utils.interp_generators as igs

class InterpAttacker():
    def __init__(self, net, sample, og_label, net_proxy=None, k=20, target=None):
        """
        net: original network
        net_proxy: secondary network used to calculate gradient of attack loss w.r.t. 
            the input sample through the saliency map (i.e. this net does not have relu acivations)
        sample: original sample
        top_k: the number of features to consider when performing top-k attack
        """
        self.net = net
        if net_proxy == None:
            self.net_proxy = net
        else:
            self.net_proxy = net_proxy
        self.og_label = og_label
        self.can_attack = self._prediction_correct(sample)
        if not self.can_attack:
            raise Exception("Network's prediction incorrect. Attacking is meaningless. Try attacking a different sample.")
        self.og_sample = torch.autograd.Variable(sample, requires_grad=True)
        self.k = k
        self.target = target
        self.c = self.og_sample.shape[-3]
        self.w = self.og_sample.shape[-2]
        self.h = self.og_sample.shape[-1]
    
    def _set_interp_generator(self, method='simple_gradient'):
        """Sets the mechanism used to create the interps.
        """
        if method == 'simple_gradient':
            self.interp_generator = igs.simple_gradient
        elif method == 'smoothgrad':
            self.interp_generator = igs.smoothgrad
        else:
            print(f'{method} is not supported')

    def _prediction_correct(self, sample):
        """If network's prediction for the input is incorrect, attacking
        has no meaning.
        """ 
        output = self.net(sample)
        return torch.argmax(output) == self.og_label
    
    def _mass_center(self, saliency):
        """Finds the coordinates of the center of mass of the saliency map.
        """
        saliency = saliency.float()
        x_mesh, y_mesh = torch.meshgrid([torch.arange(self.w), torch.arange(self.h)])
        x_mesh, y_mesh = x_mesh.float(), y_mesh.float()
        mass_center = torch.stack([torch.sum(saliency * x_mesh)/(self.w * self.h), torch.sum(saliency * y_mesh)/(self.w * self.h)])
        return mass_center
    
    def _top_k(self, saliency):
        """Returns top_k, a tensor of same shape as saliency where every location
        is zero except those top-k locations of saliency, which hold ones.
        """
        top_k_idxs = torch.argsort(saliency.reshape(self.w * self.h), descending=True)[:self.k]
        top_k = torch.zeros(self.w * self.h)
        top_k[top_k_idxs] = 1.
        return top_k.reshape(self.w, self.h)
        
    def _find_perturbation(self, sample, attack_method, invert):
        """Finds the optimal direction in which to perturb each dimension of the 
        input sample so as to alter the saliency map in the desired manner.
        """
        sample = torch.autograd.Variable(sample, requires_grad=True)
        saliency = self.interp_generator(self.net_proxy, sample, self.og_label, for_loss=True)
        if attack_method == 'top-k':
            loss = torch.sum(self.og_top_k * saliency)
        elif attack_method == 'mass-center':
            mass_center = self._mass_center(saliency)
            loss = -torch.sum((self.og_mass_center - mass_center)**2)
        elif attack_method == 'targeted':
            loss = -torch.sum(saliency * self.target)
        else:
            print('invalid attack')
        if invert:
            loss = -loss
        # norm of perturbation is always 28
        perturbation = -torch.autograd.grad(loss, sample)[0].reshape(self.c, self.w, self.h)
        return torch.sign(perturbation)

    def _apply_perturbation(self, sample, perturbation, alpha):
        """Applies the perturbation to the sample.
        """
        sample = sample + alpha * perturbation
        if self.c == 3:
            sample_clipped = sample.clamp(min=-2.8, max=2.8215)
        if self.c == 1:
            sample_clipped = sample.clamp(min=-0.4242, max=2.8215)
        return sample_clipped
    
    def _check_measure(self, sample, measure):
        """Checks to see how far the interp associated with 
        sample has moved from the original interp.
        """
        saliency = self.interp_generator(self.net_proxy, sample, self.og_label, used=False, for_loss=True)
        if measure == 'intersection':
            value = torch.sum(self._top_k(saliency) * self.og_top_k) / self.k
        elif measure == 'correlation':
            value = scipy.stats.spearmanr(saliency.flatten().cpu().detach(), self.og_saliency.flatten().cpu().detach())[0]
        elif measure == 'mass-center':
            value = torch.norm(self.og_mass_center - self._mass_center(saliency))
        else:
            print(f'{measure} is not supported as a measure')
        return value
            
        
    def iterative_attack(self, sample, attack_method, interp_to_attack, measure='correlation', alpha=1., num_iters=100, invert=False):
        """Performs the desired attack method for the desired number of iterations.
        """
        self._set_interp_generator(interp_to_attack)
        self.og_saliency = self.interp_generator(self.net, self.og_sample, self.og_label, for_loss=True)
        self.og_mass_center = self._mass_center(self.og_saliency)
        self.og_top_k = self._top_k(self.og_saliency)
                
        sample_ = sample
        best_sample = sample
        best_value = 1.
        best_i = 0
        
        for i in range(num_iters):
            perturbation = self._find_perturbation(sample_, attack_method, invert)
            candidate_sample = self._apply_perturbation(sample_, perturbation, alpha)
            if self._prediction_correct(candidate_sample):
                sample_ = candidate_sample
                value = self._check_measure(sample_, measure)
                if value < best_value:
                    best_value = value
                    best_i = i
                    best_sample = sample_
            else:
                break
        
        log_probs = self.net(best_sample)
        confidence = torch.exp(torch.max(log_probs))/torch.sum(torch.exp(log_probs))
        
        return best_sample, best_i, i, confidence
        
            