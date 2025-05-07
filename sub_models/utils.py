import torch


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
