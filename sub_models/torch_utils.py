import torch
from torch.distributions import Categorical, Distribution, OneHotCategorical
import torch.nn.functional as F


class MSEDist(Distribution):
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


class MultiOneHotCategorical(Distribution):
    """
    A batched collection of independent OneHotCategorical distributions.
    Supports input shapes: [B, L, M, K], [B, M, K], or [M, K]
    """

    arg_constraints = {}
    has_rsample = False
    support = OneHotCategorical.support

    def __init__(self, logits=None, probs=None, validate_args=None):
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of `logits` or `probs`")

        self.input_type = "logits" if logits is not None else "probs"
        tensor = logits if logits is not None else probs
        # Normalize shape: always [*batch_shape, M, K]
        if tensor.dim() < 2:
            raise ValueError("Input tensor must be at least 2D [M, K]")

        # Save shape metadata
        self._batch_shape = tensor.shape[:-2]  # Could be (), (B,), or (B, L)
        self._event_shape = tensor.shape[-2:]  # (M, K)

        # Expand input to [..., M, K]
        if self.input_type == "logits":
            self._logits = tensor
            self._probs = F.softmax(tensor, dim=-1)
        else:
            self._probs = tensor
            self._logits = torch.log(tensor + 1e-8)
        self.logits = self._logits
        self.probs = self._probs
        # Split into list of OneHotCategoricals: each is [..., K]
        self._dists = [
            OneHotCategorical(logits=self._logits[..., i, :])
            for i in range(self.event_shape[0])  # over M
        ]

        super().__init__(
            batch_shape=self.batch_shape,
            event_shape=self.event_shape,
            validate_args=validate_args,
        )

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    def sample(self, sample_shape=torch.Size()):
        # Sample independently from each distribution
        samples = [
            d.sample(sample_shape) for d in self._dists
        ]  # list of [*sample_shape, *batch_shape, K]
        return torch.stack(samples, dim=-2)  # [..., M, K]

    def log_prob(self, value):
        """
        Args:
            value: [..., M, K] one-hot matrix
        Returns:
            log_prob: [...], summed over M
        """
        if value.shape[-2:] != self.event_shape:
            raise ValueError(
                f"Expected value shape ending in {self.event_shape}, got {value.shape}"
            )

        logps = [
            d.log_prob(value[..., i, :]) for i, d in enumerate(self._dists)
        ]  # list of [...]
        stacked = torch.stack(logps, dim=-1)  # [..., M]
        return stacked.sum(dim=-1)  # [...]

    def entropy(self):
        """
        Returns:
            entropy: [...], summed over M
        """
        entropies = [d.entropy() for d in self._dists]  # list of [...]
        stacked = torch.stack(entropies, dim=-1)  # [..., M]
        return stacked.sum(dim=-1)  # [...]

    def mode(self):
        """
        Returns the mode of the distribution: one-hot with max prob for each categorical var.
        Shape: [..., M, K]
        """
        indices = torch.argmax(self._probs, dim=-1)  # [..., M]
        return F.one_hot(
            indices, num_classes=self.event_shape[1]
        ).float()  # [..., M, K]


def multi_onehot_kl(q: MultiOneHotCategorical, p: MultiOneHotCategorical):
    """
    Computes KL(q || p) for two MultiOneHotCategorical distributions.
    q, p: MultiOneHotCategorical
    Returns:
        Tensor of shape [B, L] or [B] or scalar depending on shape
    """
    # q.logits: [..., M, K]
    # Flatten to [..., M, K]
    q_log_probs = F.log_softmax(q.logits, dim=-1)
    p_log_probs = F.log_softmax(p.logits, dim=-1)

    q_probs = torch.exp(q_log_probs)

    # KL = sum_i q_i * (log q_i - log p_i)
    kl = q_probs * (q_log_probs - p_log_probs)
    kl = kl.sum(dim=-1).sum(dim=-1)  # Sum over K and M

    return kl


if __name__ == "__main__":

    def test_multi_onehot_categorical():
        """
        Test the MultiOneHotCategorical distribution.
        """
        print("\n----------------Test MultiOneHotCategorical-----------------")
        # Example usage
        logits = torch.randn(3, 16, 4, 4)  # 3 categorical variables, 5 classes each
        dist = MultiOneHotCategorical(logits=logits)
        sample = dist.sample()

        print("Sample:", sample.shape)
        print("Log Probability:", dist.log_prob(sample).shape)
        print("Entropy:", dist.entropy().shape)
        print("Mode:", dist.mode().shape)

        # Example usage
        logits = torch.randn(5, 4, 4)  # 3 categorical variables, 5 classes each
        dist = MultiOneHotCategorical(logits=logits)
        sample = dist.sample()

        print("\nSample:", sample.shape)
        print("Log Probability:", dist.log_prob(sample).shape)
        print("Entropy:", dist.entropy().shape)
        print("Mode:", dist.mode().shape)

        # Example usage
        logits = torch.randn(4, 4)  # 3 categorical variables, 5 classes each
        dist = MultiOneHotCategorical(logits=logits)
        sample = dist.sample()

        print("\nSample:", sample.shape)
        print("Log Probability:", dist.log_prob(sample))
        print("Entropy:", dist.entropy())
        print("Mode:", dist.mode().shape)

    def test_multi_onehot_kl():
        """
        Test the KL divergence between two MultiOneHotCategorical distributions.
        """
        print("\n----------------Test MultiOneHotKL-----------------")
        logits_q = torch.randn(3, 16, 4, 4)
        logits_p = torch.randn(3, 16, 4, 4)
        q = MultiOneHotCategorical(logits=logits_q)
        p = MultiOneHotCategorical(logits=logits_p)
        kl = multi_onehot_kl(q, p)
        print("KL Divergence:", kl.shape)

    test_multi_onehot_kl()
