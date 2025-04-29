import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# This implementation is similar to tf.keras.layers.MultiHeadAttention, not torch.nn.MultiheadAttention.
# This can be used in the situation where query = key = value.
# In RT-1 we don't set value_dim. Therefore, values_dim = key_dim.
class TF_MultiHeadAttention(nn.Module):
    def __init__(self, 
                 heads: int, 
                 d_model: int, 
                 key_dim: int, 
                 value_dim: Optional[int] = None, 
                 dropout: float = 0.1,
                 return_attention_scores: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.h = heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim else key_dim # if value_dim is None, value_dim will be key_dim.
        self.return_attention_scores = return_attention_scores

        self.q_linear = nn.Linear(d_model, self.h * self.key_dim)
        self.k_linear = nn.Linear(d_model, self.h * self.key_dim)
        self.v_linear = nn.Linear(d_model, self.h * self.value_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.h * self.value_dim, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.key_dim)
        q = self.q_linear(q).view(bs, -1, self.h, self.key_dim)
        v = self.v_linear(v).view(bs, -1, self.h, self.value_dim)
        
        # transpose to get dimensions bs * h * sl * key_dim or bs * h * sl * value_dim
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2) # calculate attention using function we will define next

        if self.return_attention_scores:
            attention_output, score = attention(q, k, v, self.key_dim, mask, self.dropout, self.return_attention_scores) # attention_output: (bs, h, sl, value_dim), score: (bs, h, sl, sl)
        else:
            attention_output = attention(q, k, v, self.key_dim, mask, self.dropout, self.return_attention_scores) # (bs, h, sl, value_dim)
        
        # concatenate heads and put through final linear layer
        concat = attention_output.transpose(1,2).contiguous().view(bs, -1, self.h * self.value_dim) # (bs, sl, heads * value_dim)
        output = self.out(concat) # (bs, sl, d_model)

        if self.return_attention_scores:
            return output, score
        else:
            return output


def attention(q, k, v, key_dim, mask=None, dropout=None, return_attention_scores=False):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(key_dim)
    # q: (bs, h, sl, key_dim)
    # k.transpose(-2, -1) : (bs, h, key_dim, sl)
    # score: (bs, h, sl, sl)
    
    if mask is not None:
        # mask: (sl, sl)
        mask = mask.unsqueeze(0).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)

    
    if dropout is not None:
        scores = dropout(scores)


    output = torch.matmul(scores, v)
    # score: (bs, h, sl, sl)
    # v : (bs, h, sl, value_dim)
    # output: (bs, h, sl, value_dim)

    if return_attention_scores:
        return output, scores
    else:
        return output
    
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
class FFN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.linear3 = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x) * self.linear3(x)))
    
class MoE(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, noisy_gating=True, k=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([FFN(self.input_size, hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

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
        return x.float().var() / (x.float().mean()**2 + eps)

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
        normal = torch.distributions.normal.Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, seq_len, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, seq_len, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        b, s = x.shape[0], x.shape[1]
        x = x.reshape(-1, self.input_size)
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(b, s, self.input_size)
        return y, loss

# input_size and output_size: (bs, sl, feed_forward_size)
class _TransformerLayer(nn.Module):
    """A single transformer block."""
    def __init__(self,
            layer_size: int = 4096, # This corresponds to key_dim which is the size of each attention head for query, key and values.
            num_heads: int = 8,
            feed_forward_size: int = 512, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate: float = 0.1,
            return_attention_scores: bool = False):

        super(_TransformerLayer, self).__init__()
        self._return_attention_scores = return_attention_scores

        self.norm_1 = nn.LayerNorm(feed_forward_size)
        self.attn = TF_MultiHeadAttention(num_heads,feed_forward_size, layer_size, dropout=dropout_rate, return_attention_scores=return_attention_scores)
        # self.ff = FFN(feed_forward_size, 4 * feed_forward_size)
        self.ff = MoE(feed_forward_size, 4 * feed_forward_size, num_experts=8, noisy_gating=True, k=2)
        self.norm_2 = nn.LayerNorm(feed_forward_size)
        self.dropout_1 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x1 = self.norm_1(x)
        attn_results = self.attn(x1, x1, x1, mask=mask)
        if self._return_attention_scores:
            x1, score = attn_results
        else:
            x1, score = attn_results, None
        x = x + x1

        y = self.norm_2(x)
        ff_y, moe_loss = self.ff(y)
        ff_y = self.dropout_1(ff_y)
        x = x + ff_y

        return x, score, moe_loss

class Transformer(nn.Module):
    def __init__(self,
            num_layers: int = 1, # Number of transformer layers.
            layer_size: int = 4096, # This corresponds to key_dim which is the size of each attention head for query, key and values.
            num_heads: int = 8,
            feed_forward_size: int = 512, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate: float = 0.1,
            vocab_size: int = 256, # Dimensionality of tokens from the output layer. This is also dimensionality of tokens from the input layer.
            input_token_emb_dim: int = 512, # embedding dim of input tokens.
            return_attention_scores: bool = False,
            max_seq_len: int = 256 # Maximum sequence length. This Transformer can't receive tokens that are more than this number.
            ):
        super(Transformer, self).__init__()

        # issue#3 use nn.ModuleList
        self._layers = nn.ModuleList([
        _TransformerLayer(  # pylint: disable=g-complex-comprehension
            layer_size=layer_size,
            num_heads=num_heads,
            feed_forward_size=feed_forward_size,
            dropout_rate=dropout_rate,
            return_attention_scores=return_attention_scores)
            for _ in range(num_layers)
        ])

        self._token_emb = nn.Linear(input_token_emb_dim, feed_forward_size)
        self._position_emb = nn.Embedding(max_seq_len, feed_forward_size) # <--- 工夫が必要 ここだけBERTのようにする？
        self._output_tokens = nn.Linear(feed_forward_size, vocab_size)

    # inputs: (bs, seq, emb_dim). emb_dim = vocab_size
    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        # 1. Token Embeddings
        tokens_embeddings = self._token_emb(inputs) # (bs, seq_len, feed_forward_size)

        # 2. Transformer Positional Embedding：
        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
        position_ids = torch.tile(position_ids.unsqueeze(0), dims=(batch_size, 1)) # (bs, seq_len)
        position_embeddings = self._position_emb(position_ids) # (bs, seq_len, feed_forward_size)

        # Add the two embedded tensors together
        x = tokens_embeddings + position_embeddings # (bs, seq_len, feed_forward_size)

        scores = []
        moe_loss = 0

        for layer in self._layers:
            x, score, layer_moe_loss = layer(x, mask=attention_mask)
            if score is not None:
                scores.append(score)
            moe_loss += layer_moe_loss
        x = self._output_tokens(x) # (bs, seq_len, vocab_size)
        return x, scores, moe_loss