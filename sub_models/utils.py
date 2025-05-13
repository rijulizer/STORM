import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class MSEDist(torch.distributions.Distribution):
    """
    A custom distribution class that represents a mean squared error (MSE) distribution.
    It is used to compute the MSE loss between two tensors.
    #TODO: For now this is a dummy implementation, Check the logic in future.
    """

    def __init__(self, pred, dims, agg="sum"):
        """ """
        super().__init__()
        self.pred = pred
        self._dim = dims
        self._axes = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg

    def sample(self, sample_shape):
        """
        Generates samples by broadcasting the prediction tensor to the desired shape.
        """
        return self.pred.expand(sample_shape, self.pred.shape)

    def mode(self):
        """
        Returns the mode of the distribution, which is the prediction tensor.
        """
        return self.pred

    def log_prob(self, value):
        """
        Computes the log probability of a given value.

        Args:
            value (torch.Tensor): The value for which to compute the log probability.

        Returns:
            torch.Tensor: The log probability of the value.
        """
        assert len(self.pred.shape) == len(value.shape), (self.pred.shape, value.shape)
        distance = (self.pred - value) ** 2
        loss = distance.sum(self._axes)

        return loss


class OneHotDist(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        """
        Args:
            logits: Tensor of shape [..., M, K], where:
                - ... are any number of batch dims (e.g., B, L)
                - M is number of categorical variables
                - K is number of classes per variable
        """
        super().__init__()
        assert logits.dim() >= 2, "Expected logits of shape [..., M, K]"
        self.logits = logits
        *self.B, self.M, self.K = logits.shape

    def sample(self):
        """
        Returns:
            Differentiable one-hot sample of shape [..., M, K]
        """
        # Flatten for sampling: [B*L*M, K]
        flat_logits = self.logits.reshape(-1, self.K)  # [N*M, K]
        indices = Categorical(logits=flat_logits).sample()  # [N*M]
        hard = F.one_hot(indices, num_classes=self.K).float()  # [N*M, K]
        hard = hard.reshape(*self.B, self.M, self.K)

        probs = F.softmax(self.logits, dim=-1)
        return (hard - probs).detach() + probs  # Straight-through gradient trick

    def log_prob(self, one_hot_action: torch.Tensor):
        """
        Args:
            one_hot_action: [..., M, K] one-hot encoded action

        Returns:
            log_prob: [..., M]
        """
        log_probs = F.log_softmax(self.logits, dim=-1)
        return (log_probs * one_hot_action).sum(dim=-1)

    def entropy(self):
        """
        Computes the entropy of the distribution.
        E = -sum(p * log(p)) where p is the probability of each class.
        Returns:
            torch.Tensor: The entropy of the distribution.
        """
        log_probs = F.log_softmax(self.logits, dim=-1)
        probs = torch.exp(log_probs)
        return -(log_probs * probs).sum(dim=-1)
