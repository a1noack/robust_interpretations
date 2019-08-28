import torch

class SimpleGradientsAttack():
    def __init__(self, net, net_proxy, sample, og_label, top_k=20, target=None, device=None):
        """
        net: original network
        net_proxy: secondary network used to calculate gradient of attack loss w.r.t. 
            the input sample through the saliency map (i.e. this net does not have relu acivations)
            non-relu activations)
        sample: original sample
        top_k: the number of features to consider when performing top-k attack
        device: the number of the gpu that is being used
        """
        self.net = net
        self.net_proxy = net_proxy
        self.og_label = og_label
        if not self.prediction_correct(sample):
            print("Network's prediction incorrect. Attacking is meaningless.")
            return
        self.og_sample = torch.autograd.Variable(sample, requires_grad=True)
        self.top_k = top_k
        self.target = target
        self.device = device
        self.og_saliency = self.create_saliency_map(self.net, self.og_sample)
        self.h = self.og_saliency.shape[-1]
        self.w = self.og_saliency.shape[-2]
        self.og_mass_center = self.find_mass_center(self.og_saliency)

    def prediction_correct(self, sample):
        """If network's prediction for the input is incorrect, attacking
        has no meaning.
        """ 
        output = self.net(sample)
        return torch.argmax(output) == self.og_label
    
    def create_saliency_map(self, net, sample, used=True):
        """Creates saliency map for sample.
        """
        if not used:
            sample = torch.autograd.Variable(sample, requires_grad=True)
        net(sample)
        logits = net.logits
        bp_mat = torch.nn.functional.one_hot(self.og_label, num_classes=10).float()
        if self.device != None:
            bp_mat = bp_mat.to(self.device)
        logit = torch.sum(bp_mat * logits)
        grads = torch.autograd.grad(logits * bp_mat, sample, grad_outputs=bp_mat, create_graph=True)[0].squeeze()
        saliency = torch.abs(grads)
        saliency = saliency/torch.sum(saliency)
        # equivalent to 
        # torch.autograd.grad(logit, sample)[0].squeeze()
        return saliency
    
    def find_mass_center(self, saliency):
        """Finds the coordinates of the center of mass of the saliency map.
        """
        saliency = saliency.float()
        x_mesh, y_mesh = torch.meshgrid([torch.arange(self.w), torch.arange(self.h)])
        x_mesh, y_mesh = x_mesh.float(), y_mesh.float()
        mass_center = torch.stack([torch.sum(saliency * x_mesh)/(self.w * self.h),torch.sum(saliency * y_mesh)/(self.w * self.h)])
        return mass_center
        
    def find_perturbation(self, sample, attack_method):
        """Finds the optimal direction in which to perturb each dimension of the 
        input sample so as to alter the saliency map in the desired manner.
        """
        sample = torch.autograd.Variable(sample, requires_grad=True)
        saliency = self.create_saliency_map(self.net_proxy, sample)
        if attack_method == 'top_k':
            top_k_idxs = torch.argsort(saliency.reshape(self.w * self.h), descending=True)[:self.top_k]
            top_k_elems = torch.zeros(self.w * self.h)
            top_k_elems[top_k_idxs] = 1
            top_k_elems = top_k_elems * saliency.reshape(self.w * self.h)
            top_k_loss = torch.sum(top_k_elems)
            loss = top_k_loss
        elif attack_method == 'mass_center':
            mass_center = self.find_mass_center(saliency)
            mass_center_loss = torch.norm(self.og_mass_center - mass_center)
            loss = -mass_center_loss
        elif attack_method == 'targeted':
            targeted_loss = torch.sum(saliency * self.target)
            loss = -targeted_loss
        else:
            print('invalid attack')
        # norm of perturbation is always 28
        perturbation = -torch.autograd.grad(loss, sample)[0]
        return torch.sign(perturbation.reshape((1, self.w, self.h)))

    def apply_perturbation(self, sample, perturbation, alpha):
        """Applies the perturbation to the sample.
        """
        sample = sample + alpha * perturbation
        return sample
        
    def iterative_attack(self, sample, attack_method, alpha=1., num_iters=100):
        """Performs the desired attack method for the desired number of iterations.
        """
        for i in range(num_iters):
            perturbation = self.find_perturbation(sample, attack_method)
            if i % 50 == 0:
                print(f'iteration #{i}')
            candidate = self.apply_perturbation(sample, perturbation, alpha)
            if self.prediction_correct(candidate):
                sample = candidate
            else:
                break
        return sample
        
        
        
        
        
            