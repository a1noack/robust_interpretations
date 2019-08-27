import torch

class SimpleGradientsAttack():
    def __init__(self, net, net_proxy, sample, og_label, top_k=20, device=None):
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
        self.device = device
        self.og_saliency = self.create_saliency_map(self.net_proxy, self.og_sample)

    def prediction_correct(self, sample):
        """If network's prediction for the input is incorrect, attacking
        has no meaning.
        """ 
        output = self.net(sample)
        return (torch.argmax(output).item() == self.og_label.item())
    
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
        grads = torch.autograd.grad(logits*bp_mat, sample, grad_outputs=bp_mat, create_graph=True)[0].squeeze()
        saliency = torch.abs(grads)
        saliency = saliency/torch.sum(saliency)
        # equivalent to 
        # torch.autograd.grad(logit, sample)[0].squeeze()
        return saliency
    
    def find_perturbation(self, sample, attack_method):
        sample = torch.autograd.Variable(sample, requires_grad=True)
        saliency = self.create_saliency_map(self.net_proxy, sample)
        h = saliency.shape[-1]
        w = saliency.shape[-2]
        if attack_method == 'top_k':
            top_k_idxs = torch.argsort(saliency.reshape(w*h), descending=True)[:self.top_k]
            top_k_elems = torch.zeros(w*h)
            top_k_elems[top_k_idxs] = 1
            top_k_elems = top_k_elems * saliency.reshape(w*h)
            top_k_loss = torch.sum(top_k_elems)
            perturbation = -torch.autograd.grad(top_k_loss, sample)[0]
            perturbation = torch.sign(perturbation.reshape((1, w, h)))
        return perturbation

    def apply_perturbation(self, sample, perturbation, alpha):
        sample = sample + alpha * perturbation
        return sample
        
    def iterative_attack(self, sample, attack_method, alpha=1, num_iters=100):
        for i in range(num_iters):
            perturbation = self.find_perturbation(sample, attack_method)
            if i % 5 == 0:
                print(f'iteration #{i}. norm of perturbation = {torch.norm(perturbation)}')
            candidate = self.apply_perturbation(sample, perturbation, alpha)
            if self.prediction_correct(candidate):
                sample = candidate
            else:
                break
        return sample
        
        
        
        
        
            