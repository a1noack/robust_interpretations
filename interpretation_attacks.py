import torch

class SimpleGradientsAttack():
    def __init__(self, net, sample, og_label, top_k=20):
        if not self.prediction_correct(net, sample, og_label):
            print("Network's prediction incorrect. Attacking is meaningless.")
            return
        self.net = net
        self.sample = torch.autograd.Variable(sample, requires_grad=True)
        self.og_label = og_label

    def prediction_correct(self, net, sample, og_label):
        """If network's prediction for the input is incorrect, attacking
        has no meaning.
        """ 
        output = net(sample)
        return (torch.argmax(output).item() == og_label.item())
    
    def saliency_map(self):
        """Creates saliency map given a pytorch model and a sample.
        """
        output = self.net(self.sample)
        bp_mat = torch.zeros(output.shape)
        bp_mat[0, torch.argmax(output)] = 1.
        return torch.autograd.grad(output, self.sample, grad_outputs=bp_mat)[0].squeeze()
    def predict(self, vector=True):
        """Returns prediction for self.sample.
        """
        output = self.net(self.sample)
        if vector:
            return output
        return torch.argmax(output).item()