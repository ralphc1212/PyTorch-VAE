import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math

def scaled_dot_product(q, k, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attention = F.softmax(attn_logits, dim=-1)
    return attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 2 * embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k = qkv.chunk(2, dim=-1)

        # Determine value outputs
        attention = scaled_dot_product(q, k, mask=mask)

        return attention


TAU = 1.
PI = 0.95
RSV_DIM = 4
EPS = 1e-8
SAMPLE_LEN = 0.25

class VNDAE_ATTN(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VNDAE_ATTN, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.msa = MultiheadAttention(input_dim=4, embed_dim=1, num_heads=1)

        Pi = nn.Parameter(PI * torch.ones(latent_dim - RSV_DIM), requires_grad=False)

        self.ZERO = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.ONE = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.pv = nn.Parameter(torch.cat([self.ONE, torch.cumprod(Pi, dim=0)])
                       * torch.cat([1 - Pi, self.ONE]), requires_grad=False)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    @staticmethod
    def clip_beta(tensor, to=5.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)
        result_1 = torch.flatten(result, start_dim=1)
        result_2 = result.view(*result.shape[:2], -1)

        mu = self.fc_mu(result_1)
        log_var = self.fc_var(result_1)
        attention = self.msa(result_2)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        p_vnd = torch.diagonal(attention.squeeze(1), 0, 1, 2)

        return [mu, log_var, p_vnd]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor, p_vnd: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        beta = p_vnd[:,RSV_DIM:]
        ONES = torch.ones_like(beta[:,0:1])
        qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)
        s_vnd = F.gumbel_softmax(qv, tau=TAU, hard=True)

        cumsum = torch.cumsum(s_vnd, dim=1)
        dif = cumsum - s_vnd
        mask0 = dif[:, 1:]
        mask1 = 1. - mask0
        s_vnd = torch.cat([torch.ones_like(p_vnd[:,:RSV_DIM]), mask1], dim = -1)

        return (eps * std + mu) * s_vnd

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, p_vnd = self.encode(input)
        z = self.reparameterize(mu, log_var, p_vnd)
        return  [self.decode(z), input, mu, log_var, p_vnd]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        p_vnd = args[4]
        beta = p_vnd[:,RSV_DIM:]
        ONES = torch.ones_like(beta[:,0:1])
        qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)

        ZEROS = torch.zeros_like(beta[:, 0:1])
        cum_sum = torch.cat([ZEROS, torch.cumsum(qv[:, 1:], dim = 1)], dim = -1)[:, :-1]
        coef1 = torch.sum(qv, dim=1, keepdim=True) - cum_sum
        coef1 = torch.cat([torch.ones_like(p_vnd[:,:RSV_DIM]), coef1], dim = -1)

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_gaussian = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())

        kld_weighted_gaussian = torch.diagonal(kld_gaussian.mm(coef1.t()), 0).mean()

        log_frac = torch.log(qv / self.pv + EPS)
        kld_vnd = torch.diagonal(qv.mm(log_frac.t()), 0).mean()

        kld_loss = kld_vnd + kld_weighted_gaussian
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD': - kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = torch.cat([z[:, :int(SAMPLE_LEN * self.latent_dim)], torch.zeros_like(z[:, :int((1 - SAMPLE_LEN) * self.latent_dim)])], dim = -1)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]