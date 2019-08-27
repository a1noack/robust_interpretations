import torch

class SimpleGradientsAttack():
    def __init__(self, net, sample, og_label, top_k=20, device=None):
        if not self.prediction_correct(net, sample, og_label):
            print("Network's prediction incorrect. Attacking is meaningless.")
            return
        self.net = net
        self.sample = torch.autograd.Variable(sample, requires_grad=True)
        self.og_label = og_label
        self.top_k = top_k
        self.device = device
        self.saliency = self.create_saliency_map()

    def prediction_correct(self, net, sample, og_label):
        """If network's prediction for the input is incorrect, attacking
        has no meaning.
        """ 
        output = net(sample)
        return (torch.argmax(output).item() == og_label.item())
    
    def create_saliency_map(self):
        """Creates saliency map given a pytorch model and a sample.
        """
        self.net(self.sample)
        logits = self.net.logits
        bp_mat = torch.nn.functional.one_hot(self.og_label, num_classes=10).float()
        if self.device != None:
            bp_mat = bp_mat.to(self.device)
        logit = torch.sum(bp_mat * logits)
        grads = torch.autograd.grad(logits*bp_mat, self.sample, grad_outputs=bp_mat, create_graph=True)[0].squeeze()
        saliency = torch.abs(grads)
        saliency = saliency/torch.sum(saliency)
        # equivalent to 
        # torch.autograd.grad(logit, self.sample)[0].squeeze()
        return saliency
    
    def find_perturbation(self, sample_, attack_method):
        h = self.saliency.shape[-1]
        w = self.saliency.shape[-2]
        if attack_method = 'top_k':
            top_k_idxs = torch.argsort(w*h, descending=True)[:self.top_k]
            top_k_elems = torch.zeros(w*h)
            top_k_elems[top_k_idxs] = 1
            top_k_elems = top_k_idxs * self.saliency.reshape(w*h)
            top_k_loss = torch.sum(top_k_elems)
            self.perturbation = torch.autograd.grad(top_k_loss, sample_)
            self.perturbation = torch.sign(self.perturbation.reshape((1, w, h)))
    
    def apply_perturbation(self, sample_, alpha):
        sample_ = sample_ + alpha * self.perturbation
        return sample_
        
    def iterative_attack(self, sample_, attack_method, alpha=1, num_iters=100):
        for i in range(num_iters):
            self.find_perturbation(sample_, attack_method)
            sample_ = self.apply_perturbation(sample_, alpha)
        return sample_
        
        
        
        
        
            