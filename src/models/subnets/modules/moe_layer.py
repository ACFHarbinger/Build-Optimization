"""Mixture of Experts (MoE) layer."""

import torch
from torch import nn
from torch.distributions.normal import Normal

from .moe_dispatcher import SparseDispatcher


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
        input_size: integer - size of the input
        output_size: integer - size of the input
        num_experts: an integer - number of experts
        num_neurons: a list - hidden dimension of the experts
        noisy_gating: a boolean
        k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_neurons=None,
        experts=None,
        hidden_act="ReLU",
        out_bias=True,
        num_experts=4,
        k=2,
        noisy_gating=True,
        **kwargs,
    ):
        """
        Initialize the MoE layer.

        Args:
            input_size (int): Dimension of input features.
            output_size (int): Dimension of output features.
            num_neurons (list, optional): Hidden dimensions for MLP experts.
            experts (list, optional): List of expert modules. If None, builds MLPs.
            hidden_act (str): Activation function name for MLP experts.
            out_bias (bool): Whether to include bias in output layer.
            num_experts (int): Number of experts to create.
            k (int): Number of experts to select per input.
            noisy_gating (bool): Whether to use noisy gating for load balancing.
            **kwargs: Extra arguments.
        """
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        # instantiate experts
        if experts is not None:
            self.experts = experts
        elif num_neurons is not None and len(num_neurons) > 0:
            # Basic MLP implementation if not provided
            def build_mlp(dims):
                """Builds a Multi-Layer Perceptron expert."""
                layers = []
                in_dim = input_size
                for hidden in dims:
                    layers.append(nn.Linear(in_dim, hidden))
                    layers.append(getattr(nn, hidden_act)())
                    in_dim = hidden
                layers.append(nn.Linear(in_dim, output_size, bias=out_bias))
                return nn.Sequential(*layers)

            self.experts = nn.ModuleList([build_mlp(num_neurons) for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList(
                [nn.Linear(self.input_size, self.output_size, bias=out_bias) for _ in range(self.num_experts)]
            )

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
            x: a `Tensor`.
        Returns:
            a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
            gates: a `Tensor` of shape [batch_size, n]
        Returns:
            a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
            clean_values: a `Tensor` of shape [batch, n].
            noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
              normally distributed noise with standard deviation noise_stddev.
            noise_stddev: a `Tensor` of shape [batch, n], or None
            noisy_top_values: a `Tensor` of shape [batch, m].
               "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
            a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: <https://arxiv.org/abs/1701.06538>.
        Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
        Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        noisy_logits = None
        noise_stddev = None
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  # non-topk elements will be 0

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=0.0):
        """
        Token/Node-level Gating with the default gating algorithm in <https://arxiv.org/abs/1701.06538>.
        In specific, each token/node chooses TopK experts, auxiliary losses required for load balancing.
        Empirically, we found that the load-balancing loss may conflict with the reinforcement loss,
        and thus we do not return loss (i.e., loss_coef=0.).
        Please Refer to (Zhou et al, 2024) <https://arxiv.org/abs/2405.01029>.

        Args:
            x: tensor shape [batch_size, problem_size, input_size]
            loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
            y: a tensor with shape [batch_size, problem_size, output_size].
            loss: a scalar. This should be added into the overall training loss of the model.
                The backpropagation of this loss encourages all experts to be approximately equally used across a batch.
        """
        output_shape = list(x.size()[:-1]) + [self.output_size]
        x = x.reshape(-1, self.input_size) if x.dim() != 2 else x

        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        return y.reshape(output_shape)
