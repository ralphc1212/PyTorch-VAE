import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

RSV_DIM = 1
SAMPLE_LEN = 1.

bs = 128
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


class VNDAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VNDAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    @staticmethod
    def clip_beta(tensor, to=5.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h), self.fc33(h) # mu, log_var

    def sampling(self, mu, log_var, p_vnd):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)\

        beta = torch.sigmoid(self.clip_beta(p_vnd[:,RSV_DIM:]))
        ONES = torch.ones_like(beta[:,0:1])
        qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)
        s_vnd = F.gumbel_softmax(qv, tau=TAU, hard=True)

        cumsum = torch.cumsum(s_vnd, dim=1)
        dif = cumsum - s_vnd
        mask0 = dif[:, 1:]
        mask1 = 1. - mask0
        s_vnd = torch.cat([torch.ones_like(p_vnd[:,:RSV_DIM]), mask1], dim = -1)

        return (eps.mul(std).add_(mu)).mul(s_vnd) # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var, p_vnd = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var, p_vnd)
        return self.decoder(z), mu, log_var, p_vnd

# build model
vae = VNDAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=4)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var, p_vnd):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    kld_gaussian = - 0.5 * (1 + log_var - mu ** 2 - log_var.exp())

    beta = torch.sigmoid(self.clip_beta(p_vnd[:,RSV_DIM:]))
    ONES = torch.ones_like(beta[:,0:1])
    qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)

    ZEROS = torch.zeros_like(beta[:, 0:1])
    cum_sum = torch.cat([ZEROS, torch.cumsum(qv[:, 1:], dim = 1)], dim = -1)[:, :-1]
    coef1 = torch.sum(qv, dim=1, keepdim=True) - cum_sum
    coef1 = torch.cat([torch.ones_like(p_vnd[:,:RSV_DIM]), coef1], dim = -1)

    kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

    kld_weighted_gaussian = torch.diagonal(kld_gaussian.mm(coef1.t()), 0).mean()

    log_frac = torch.log(qv / self.pv + EPS)
    kld_vnd = torch.diagonal(qv.mm(log_frac.t()), 0).mean()
    kld_loss = kld_vnd + kld_weighted_gaussian

    loss = BCE + 0.00025 * kld_loss

    return loss

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var, p_vnd = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var, p_vnd)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var, p_vnd = vae(data)

            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var, p_vnd).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, 51):
    train(epoch)
    test()

# with torch.no_grad():
#     z = torch.randn(64, 4).cuda()
#     sample = vae.decoder(z).cuda()

#     save_image(sample.view(64, 1, 28, 28), './mnist_samples/sample_vnd' + '.png')

