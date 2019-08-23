import torch

class SimpleGradientsAttack():
    def __init__(self, net, sample, og_label, top_k=20):
        if not self.prediction_correct():
            print("Network's prediction incorrect. Attacking is meaningless.")
            return
        self.net = net
        self.sample = sample
        self.og_label = og_label

    def prediction_correct(self):
        """If network's prediction for the input is incorrect, attacking
        has no meaning.
        """ 
        output = self.net(self.sample)
        return (torch.argmax(output).item() == self.og_label.item())
    
    def saliency_map(self):
        """Creates saliency map given a pytorch model and a sample.
        """
        output = self.net(self.sample)
        bp_mat = torch.zeros(output.shape)
        bp_mat[torch.argmax(output)] = 1.
        return torch.autograd.grad(output, self.sample, grad_outputs=bp_mat)
        