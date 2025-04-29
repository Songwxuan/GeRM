import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionQuantizer(nn.Module):
    def __init__(self, 
                 action_dim,
                 condition_dim,
                 action_ranges,
                 condition_norms,
                 num_embeddings=256, 
                 embedding_dim=16,
                 encoder_hidden_dims=[512, 256],
                 activation='ELU',
                 commitment_cost=0.25):
        super().__init__()
        activation = get_activation(activation)

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(condition_dim + action_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], embedding_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(condition_dim + embedding_dim, encoder_hidden_dims[-1]))
        decoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l - 1]))
            decoder_layers.append(activation)
        decoder_layers.append(nn.Linear(encoder_hidden_dims[0], action_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, input):
        output = self.encode(input)
        output = self.decode(output)
        reconstruction_loss = F.mse_loss(output['reconstructed_actions'], output['actions'])
        output.update({
            'reconstruction_loss': reconstruction_loss
        })
        return output
    
    def encode(self, input):
        x = torch.cat([input['actions'], input['conditions']], dim=-1)
        z = self.encoder(x)
        output = self.quantizer(z)
        output.update(input)
        return output
    
    def decode(self, output):
        input = torch.cat([output['quantized_latents'], output['conditions']], dim=-1)
        reconstructed_actions = self.decoder(input)
        output.update({
            "reconstructed_actions": reconstructed_actions
        })
        return output
    
    def decode_from_index(self, encoding_index, conditions):
        quantized_latents = self.quantizer._embedding(encoding_index)
        output = {
            'quantized_latents': quantized_latents
        }
        return self.decode(output, conditions)
    

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 distance='cos', anchor='probrandom', first_batch=False, 
                 contras_loss=True):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._pool = FeaturePool(self._num_embeddings, self._embedding_dim)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self.register_buffer("embed_prob", torch.zeros(self._num_embeddings))
        self._commitment_cost = commitment_cost

        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

    def forward(self, inputs):
        input_shape = inputs.shape
        
        # Calculate distances
        if self.distance == 'l2':
            distances = -torch.sum(inputs.detach()**2, dim=-1, keepdim=True) - \
                        torch.sum(self._embedding.weight**2, dim=-1) + \
                        2 * torch.matmul(inputs.detach(), self._embedding.weight.t())
        elif self.distance == 'cos':
            normed_inputs = F.normalize(inputs, dim=-1).detach()
            normed_weights = F.normalize(self._embedding.weight, dim=-1)
            distances = torch.matmul(normed_inputs, normed_weights.t())

            
        # Encoding
        encoding_indices = torch.argmax(distances, dim=-1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized_latents = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = self._commitment_cost * F.mse_loss(quantized_latents.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized_latents, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized_latents = inputs + (quantized_latents - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        if self.training:
            # Calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # Running average updates
            if self.anchor in ['closest', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = distances.sort(dim=0)
                    random_feat = inputs.detach()[indices[-1, :]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self._pool.query(inputs.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = torch.softmax(distances.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = inputs.detach()[prob]
                decay = torch.exp(-(self.embed_prob * self._num_embeddings * 10) / (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self._embedding_dim)
                self._embedding.weight.data = self._embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            
            # Contrastive loss
            if self.contras_loss:
                sort_distance, indices = distances.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self._num_embeddings)):, :].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0) / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
            else:
                contra_loss = 0.

        output = {
            'quantized_latents': quantized_latents,
            'encoding_indices': encoding_indices,
            'q_latent_loss': q_latent_loss,
            'e_latent_loss': e_latent_loss,
            'contrastive_loss': contra_loss,
            'perplexity': perplexity,
            'min_encodings': min_encodings,
        }
        return output
    
    
class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features


def get_activation(activation='ELU'):
    if activation == 'ELU':
        return nn.ELU()
    elif activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation == 'Tanh':
        return nn.Tanh()
    elif activation == 'Sigmoid':
        return nn.Sigmoid()
    elif activation == 'Softplus':
        return nn.Softplus()
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    model = ActionQuantizer(12, 12, 256, 16, [512, 256], activation='ELU')
    print(model)
    a = torch.randn(1, 12)
    c = torch.randn(1, 12)
    input = {
        'actions': a,
        'conditions': c
    }
    output = model(input)
    print(output.keys())
    # print(output['reconstructed_actions'].shape)
