import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
#from resnet18 import ResNet18Enc, ResNet18Dec

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BottleneckEnc(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

class ResNet50Enc(nn.Module):

    def __init__(self, num_Blocks=[3,4,6,3], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BottleneckEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BottleneckEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BottleneckEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BottleneckEnc, 512, num_Blocks[3], stride=2)
        self.linear1 = nn.Linear(2048, z_dim)
        self.linear2 = nn.Linear(2048, z_dim)

    def _make_layer(self, block, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [block(self.in_planes, planes, stride)]
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        mu = self.linear1(x)
        logvar = self.linear2(x)

        return mu, logvar

class BottleneckDec(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv3 = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, int(planes / self.expansion),
                #           kernel_size=1, stride=stride, bias=False),
                ResizeConv2d(planes * self.expansion, planes, kernel_size=3, scale_factor=self.expansion),
                nn.BatchNorm2d(planes)
            )

            self.conv2 = ResizeConv2d(planes * self.expansion, planes, kernel_size=3, scale_factor=self.expansion)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.shortcut = nn.Sequential()
            self.conv2 = nn.Conv2d(planes * self.expansion, planes, kernel_size=3)
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.leaky_relu(self.bn3(self.conv3(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

class ResNet50Dec(nn.Module):

    def __init__(self, num_Blocks=[3,6,4,3], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 2048

        self.linear = nn.Linear(z_dim, 2048)

        self.layer4 = self._make_layer(BottleneckDec, 512, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BottleneckDec, 256, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BottleneckDec, 128, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BottleneckDec, 64, num_Blocks[0], stride=1)
        self.conv1 = nn.ConvTranspose2d(64, nc, kernel_size=3, output_padding=1)

    def _make_layer(self, block, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [block(self.in_planes, planes, stride)]
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 2048, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x


class RESNET50VAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(RESNET50VAE, self).__init__()

        self.latent_dim = latent_dim

        #modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        '''
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        '''
        #self.encoder = nn.Sequential(*modules)
        self.encoder = ResNet50Enc(z_dim=latent_dim)
        #self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        #modules = []

        #self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        '''
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
        '''


        #self.decoder = nn.Sequential(*modules)
        self.decoder = ResNet50Dec(z_dim=latent_dim)
        '''
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
        '''
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        #mu = self.fc_mu(result)
        #log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        #result = self.decoder_input(z)
        #print(result.shape)
        #result = result.view(-1, 512, 2, 2)
        #result = self.decoder(result)
        result = self.decoder(z)
        #result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

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

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

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
