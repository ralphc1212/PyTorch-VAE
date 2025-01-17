import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

TAU = 1.
PI = 0.8
RSV_DIM = 1
EPS = 1e-8
SAMPLE_LEN = 1.

LATENT = 4

bs = 64
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

        Pi = nn.Parameter(PI * torch.ones(z_dim - RSV_DIM), requires_grad=False)

        self.ZERO = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.ONE = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.pv = nn.Parameter(torch.cat([self.ONE, torch.cumprod(Pi, dim=0)])
                       * torch.cat([1 - Pi, self.ONE]), requires_grad=False)

    @staticmethod
    def clip_beta(tensor, to=5.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    def loss_function(self, recon_x, x, mu, log_var, p_vnd):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='mean')
        kld_gaussian = - 0.5 * (1 + log_var - mu ** 2 - log_var.exp())

        beta = torch.sigmoid(self.clip_beta(p_vnd[:,RSV_DIM:]))
        ONES = torch.ones_like(beta[:,0:1])
        qv = torch.cat([ONES, torch.cumprod(beta, dim=1)], dim = -1) * torch.cat([1 - beta, ONES], dim = -1)

        ZEROS = torch.zeros_like(beta[:, 0:1])
        cum_sum = torch.cat([ZEROS, torch.cumsum(qv[:, 1:], dim = 1)], dim = -1)[:, :-1]
        coef1 = torch.sum(qv, dim=1, keepdim=True) - cum_sum
        coef1 = torch.cat([torch.ones_like(p_vnd[:,:RSV_DIM]), coef1], dim = -1)

        kld_weighted_gaussian = torch.diagonal(kld_gaussian.mm(coef1.t()), 0).mean()

        log_frac = torch.log(qv / self.pv + EPS)
        kld_vnd = torch.diagonal(qv.mm(log_frac.t()), 0).mean()
        kld_loss = kld_vnd + kld_weighted_gaussian

        loss = BCE + 1e-4 * kld_loss

        return loss


    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h), self.fc33(h) # mu, log_var

    def sampling(self, mu, log_var, p_vnd):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

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
vae = VNDAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=LATENT)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters(), lr=0.001)
# return reconstruction error + KL divergence losses
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var, p_vnd = vae(data)
        loss = vae.loss_function(recon_batch, data, mu, log_var, p_vnd)

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
            test_loss += vae.loss_function(recon, data, mu, log_var, p_vnd).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


BEST = 100
for epoch in range(1, 51):
    train(epoch)
    te_loss = test()
    scheduler.step()
    if te_loss < BEST:
        print('...... SAVING CHECKPOINT ......')
        torch.save(vae.state_dict(), 'mnist_model.pt')
        BEST = te_loss

vae.load_state_dict(torch.load('mnist_model.pt'))

# torch.manual_seed(63723)
# generate samples
with torch.no_grad():
    ks = [[0], [0,1], [0,1], [0,1]]
    z = {}

    name_set = set()
    for i, k in enumerate(ks):
        name_set_c = name_set.copy()
        for j, ink in enumerate(k):
            z['{}-{}'.format(str(i), str(j))] = torch.randn(64, 1).cuda()
            if i == 0:
                name_set.add('0')
            else:
                for name in name_set_c:
                    name_set.add(name+str(j))

        # for name in name_set.copy():
        #     if len(name) < i+1:
        #         name_set.remove(name)

    zero = torch.zeros(64, 1).cuda()

    for name in name_set:
        z_ = []
        for i, c in enumerate(name):
            z_.append(z['{}-{}'.format(str(i), str(c))])
        if len(name) < 4:
            z_.append(zero.repeat(1, 4 - len(name)))

        z_ = torch.cat(z_, dim=1)

        sample_1 = vae.decoder(z_).cuda()
        save_image(sample_1.view(64, 1, 28, 28), './mnist_samples/sample_vnd_'+ name + '.png')



    # for len_ in range(LATENT):
    #     l_ = len_ + 1
    #     z_ = torch.cat([z[:, :l_], torch.zeros_like(z[:, :LATENT - l_])], dim = -1)

    #     sample_1 = vae.decoder(z_).cuda()

    #     onehot = torch.nn.functional.one_hot(torch.tensor([len_]), num_classes=LATENT).to(z.device)

    #     z_ = z * onehot

    #     sample_2 = vae.decoder(z_).cuda()

    #     save_image(sample_1.view(64, 1, 28, 28), './mnist_samples/sample_vnd_len_'+ str(len_) + '.png')
    #     save_image(sample_2.view(64, 1, 28, 28), './mnist_samples/sample_vnd_dim_'+ str(len_) + '.png')




