import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

TAU = 1.
PI = 0.95
RSV_DIM = 1
EPS = 1e-8

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 conv_enc: None,
                 conv_p_vnd: None,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        Pi = nn.Parameter(PI * torch.ones(embedding_dim - RSV_DIM), requires_grad=False)

        self.ZERO = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.ONE = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.pv = nn.Parameter(torch.cat([self.ONE, torch.cumprod(Pi, dim=0)])
                       * torch.cat([1 - Pi, self.ONE]), requires_grad=False)

        self.conv_enc = conv_enc
        self.conv_p_vnd = conv_p_vnd
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    @staticmethod
    def clip_beta(tensor, to=5.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def reparameterize(self, mu: Tensor, p_vnd: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        beta = torch.sigmoid(self.clip_beta(p_vnd[:,:,:,RSV_DIM:]))
        ONES = torch.ones_like(beta[:,:,:,0:1])
        qv = torch.cat([ONES, torch.cumprod(beta, dim=-1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)
        s_vnd = F.gumbel_softmax(qv, tau=TAU, hard=True)

        cumsum = torch.cumsum(s_vnd, dim=-1)
        dif = cumsum - s_vnd
        mask0 = dif[:, :, :, 1:]
        mask1 = 1. - mask0
        s_vnd = torch.cat([torch.ones_like(p_vnd[:,:,:,:RSV_DIM]), mask1], dim = -1)

        ZEROS = torch.zeros_like(beta[:,:,:,0:1])
        cum_sum = torch.cat([ZEROS, torch.cumsum(qv[:, 1:], dim = -1)], dim = -1)[:, :-1]
        coef1 = torch.sum(qv, dim=-1, keepdim=True) - cum_sum
        coef1 = torch.cat([torch.ones_like(p_vnd[:,:RSV_DIM]), coef1], dim = -1)

        log_frac = torch.log(qv / self.pv + EPS)
        kld_vnd = torch.diagonal(qv.mm(log_frac.t()), 0).mean()

        return mu * s_vnd, kld_vnd

    def forward(self, emb: Tensor) -> Tensor:

        feat = self.conv_enc(emb)
        p_vnd = self.conv_p_vnd(emb)

        feat = feat.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        p_vnd = p_vnd.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]

        latents, kld_vnd = self.reparameterize(feat, p_vnd)

        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        # interesting
        vq_loss = (commitment_loss + kld_vnd) * self.beta + embedding_loss 

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VNDVQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VNDVQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, embedding_dim,
        #                   kernel_size=1, stride=1),
        #         nn.LeakyReLU())
        # )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        conv_enc = nn.Sequential(
                                                        nn.Conv2d(in_channels, embedding_dim,
                                                                  kernel_size=1, stride=1),
                                                        nn.LeakyReLU()),
                                        conv_p_vnd = nn.Sequential(
                                                        nn.Conv2d(in_channels, embedding_dim,
                                                                  kernel_size=1, stride=1),
                                                        nn.LeakyReLU()),
                                        beta = self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        emb = self.encoder(input)

        return [emb]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]