from .base import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *
from .wae_mmd import *
from .cvae import *
from .hvae import *
from .vampvae import *
from .iwae import *
from .dfcvae import *
from .mssim_vae import MSSIMVAE
from .fvae import *
from .cat_vae import *
from .joint_vae import *
from .info_vae import *
# from .twostage_vae import *
from .lvae import LVAE
from .logcosh_vae import *
from .swae import *
from .miwae import *
from .vq_vae import *
from .vnd_vq_vae import *
from .betatc_vae import *
from .dip_vae import *
from .vnd_ae import *
from .vnd_ae import *
from .vnd_ae_attn import *
from .odo_vae import *
from .resnet50_vae import *

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {'HVAE':HVAE,
              'LVAE':LVAE,
              'IWAE':IWAE,
              'SWAE':SWAE,
              'MIWAE':MIWAE,
              'VQVAE':VQVAE,
              'DFCVAE':DFCVAE,
              'DIPVAE':DIPVAE,
              'BetaVAE':BetaVAE,
              'InfoVAE':InfoVAE,
              'WAE_MMD':WAE_MMD,
              'VampVAE': VampVAE,
              'GammaVAE':GammaVAE,
              'MSSIMVAE':MSSIMVAE,
              'JointVAE':JointVAE,
              'BetaTCVAE':BetaTCVAE,
              'FactorVAE':FactorVAE,
              'LogCoshVAE':LogCoshVAE,
              'VanillaVAE':VanillaVAE,
              'RESNET50VAE':RESNET50VAE,
              'ConditionalVAE':ConditionalVAE,
              'CategoricalVAE':CategoricalVAE,
              'VNDAE':VNDAE,
              'VNDAE_BD':VNDAE_BD,
              'VNDVQVAE':VNDVQVAE,
              'VNDAE_ATTN':VNDAE_ATTN,
              'ODOVAE':ODOVAE}