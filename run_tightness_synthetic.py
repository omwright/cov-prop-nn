import sys, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import network_moments.torch as nm
gnm = nm.gaussian.affine_relu_affine

EPSILON = 1e-9

def standard_gaussian(x):
    return (1./np.sqrt(2*np.pi)) * torch.exp(-x*x / 2.0)

def gaussian_cdf(x):
    return 0.5 * (1.0 + torch.erf(x * (1./np.sqrt(2))))

def softrelu(x):
    return standard_gaussian(x) + x*gaussian_cdf(x)

def relu_covariance(mean, cov, var_diag, std_diag):
    cov = torch.unsqueeze(cov, 0) # Originally written for batched
    var_diag = torch.unsqueeze(var_diag, 0)
    std_diag = torch.unsqueeze(std_diag, 0)
    m1 = torch.unsqueeze(mean/(std_diag), 2)
    m2 = torch.transpose(m1, 1, 2)
    x_mean1 = torch.unsqueeze(mean, 2)
    x_mean2 = torch.transpose(x_mean1, 1, 2)
    x_std_diag1 = torch.unsqueeze(std_diag, 2)
    x_std_diag2 = torch.transpose(x_std_diag1, 1, 2)
    s11s22 = torch.unsqueeze(var_diag, dim=2)*torch.unsqueeze(var_diag, dim=1)
    rho = cov/(torch.sqrt(s11s22))
    rho = torch.clamp(rho, -1/(1+EPSILON), 1/(1+EPSILON))

    t1 = rho * x_std_diag1 * gaussian_cdf(m1) * x_std_diag2 * gaussian_cdf(m2)
    t2 = (1./2) * rho**2 * x_std_diag1 * standard_gaussian(m1) * x_std_diag2 * standard_gaussian(m2)
    t3 = (1./6) * rho**3 * -x_mean1 * standard_gaussian(m1) * -x_mean2 * standard_gaussian(m2)
    t4 = (1./24) * rho**4 * x_std_diag1*(m1**2 - 1) * standard_gaussian(m1) * x_std_diag2*(m2**2 - 1) * standard_gaussian(m2)
    return t1 + t2 + t3 + t4

def moment_linear(mean, cov, layer):
    """Propagate moments through nn.Linear"""
    mean_out = layer(mean)
    cov_out = layer.weight@cov@layer.weight.T
    return mean_out, cov_out

def moment_conv2d(mean, cov, layer):
    """Propagate moments through nn.Conv2d
    Assume mean is (c, h, w) shape
    """
    mean_out = layer(mean)
    bias = layer.bias
    layer.bias = None # Bias doesn't apply to covariance
    b, c, h, w = mean.shape
    input_shape = (c, h, w)
    input_size = c*h*w
    b, c, h, w = mean_out.shape
    output_shape = (c, h, w)
    output_size = c*h*w
    cov_out = layer(
        layer(cov.reshape(-1, *input_shape))
        .reshape(input_size, output_size)
        .T.reshape(-1, *input_shape)
    ).reshape(output_size, output_size)
    layer.bias = bias
    return mean_out, cov_out  

def moment_relu(mean, cov, layer):
    input_shape = mean.shape
    mean = mean.reshape(1, -1)
    var_diag = torch.diag(cov)
    std_diag = torch.sqrt(var_diag)
    mean_out = std_diag*softrelu(mean/(std_diag))
    cov_out = relu_covariance(mean, cov, var_diag, std_diag)
    cov_out = cov_out.squeeze()
    mean_out = mean_out.reshape(input_shape)
    return mean_out, cov_out

def propagate_moments(mean, cov, layers, input_shape=None):
    """Propagate moments through layers of a neural network
    mean     (1 x input_size)
    cov      (input_size x input_size)
    layers   list of nn modules in order
    input_shape Tuple
    """
    with torch.no_grad():
        if input_shape:
            mean = mean.reshape(-1, *input_shape)
        for layer in layers:
            if isinstance(layer, nn.Linear):
                mean, cov = moment_linear(mean, cov, layer)
            elif isinstance(layer, nn.ReLU):
                mean, cov = moment_relu(mean, cov, layer)
            elif isinstance(layer, nn.Conv2d):
                mean, cov = moment_conv2d(mean, cov, layer)
            elif isinstance(layer, nn.Flatten):
                mean = mean.reshape(1, -1)
    return mean, cov

class MLP_A(nn.Module):
    def __init__(self, input_dim, layers):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
        )
        self.relu = nn.ReLU()
        self.m2 = nn.Sequential(
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], 1)
        ) # Partitioned for easy PL-DNN

    def forward(self, x):
        out = self.m2(self.relu(self.m1(x)))
        return out

class MLP_B(nn.Module):
    def __init__(self, input_dim, layers):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3]),
        )
        self.relu = nn.ReLU()
        self.m2 = nn.Sequential(
            nn.Linear(layers[3], layers[4]),
            nn.ReLU(),
            nn.Linear(layers[4], layers[5]),
            nn.ReLU(),
            nn.Linear(layers[5], layers[6]),
            nn.ReLU(),
            nn.Linear(layers[6], 1),
        )

    def forward(self, x):
        out = self.m2(self.relu(self.m1(x)))
        return out

class CNN_A(nn.Module):
    def __init__(self, input_dim, input_channels):
        super().__init__()
        h, w = input_dim
        input_size = h*w
        self.m1 = nn.Sequential(
            nn.Conv2d(input_channels, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()
        self.m2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10*input_size, 1)
        )
    def forward(self, x):
        out = self.m2(self.relu(self.m1(x)))
        return out

class CNN_B(nn.Module):
    def __init__(self, input_dim, input_channels):
        super().__init__()
        h, w = input_dim
        input_size = h*w
        self.m1 = nn.Sequential(
            nn.Conv2d(input_channels, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()
        self.m2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10*input_size, 1)
        )
    def forward(self, x):
        out = self.m2(self.relu(self.m1(x)))
        return out

if __name__ == "__main__":
    # Configure experiment
    n_runs = 200
    n_samples = int(7.5e4)
    seed = 42

    torch.manual_seed(seed)
    dtype = torch.float64
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 8 if cuda else 0
    print(f"Cuda = {cuda} with num_workers = {num_workers}, system version = {sys.version}")

    # Run fully-connected experiments
    for network in [MLP_A, MLP_B]:
        torch.manual_seed(seed)
        time_mc = []
        time_pl = []
        time_mp = []
        ratio_mean_pl = []
        ratio_var_pl = []
        ratio_mean_mp = []
        ratio_var_mp = []

        input_size = 100
        if network is MLP_A:
            hiddens = [100, 100, 100]
        elif network is MLP_B:
            hiddens = [100, 100, 100, 100, 100, 100, 100]
        else:
            print("Error!")
            break
        
        for i in range(n_runs):
            # Set up input distribution
            mean = torch.randn(1,input_size, dtype=dtype, device=device)
            cov = nm.utils.rand.definite(input_size, dtype=dtype, device=device, 
                                        positive=True, semi=False, norm=1.0)
            input_dist = torch.distributions.MultivariateNormal(mean, cov)
        
            # Generate random MLP
            model = network(input_size, hiddens)
            if dtype == torch.float64:
                model.double()
            model = model.to(device)
        
            # Monte Carlo estimate of output statistics
            start_t = time.time()
            samples = input_dist.sample((n_samples,))
            out_samples = model(samples).detach()
            mean_mc = torch.mean(out_samples, dim=0)
            var_mc = torch.var(out_samples, dim=0)
            end_t = time.time()
            time_mc.append(end_t - start_t)
        
            # Piecewise linear estimate of output statistics (Bibi, et al., 2018)
            start_t = time.time()
            x = mean # Linearization point
            A, c1 = nm.utils.linearize(model.m1, x)
            B, c2 = nm.utils.linearize(model.m2, model.relu(model.m1(x)).detach())
            x.requires_grad_(False)
            A.squeeze_()
            c1.squeeze_()
            B.squeeze_()
            c2.squeeze_()
            mean_pl = gnm.mean(mean, cov, A, c1, B, c2)
            var_pl = gnm.special_variance(cov, A, B)
            end_t = time.time()
            time_pl.append(end_t - start_t)
        
            # Moment propagation of output statistics
            start_t = time.time()
            layers = [module for module in model.modules()]
            mean_mp, var_mp = propagate_moments(mean, cov, layers)
            end_t = time.time()
            time_mp.append(end_t - start_t)
        
            # Compute ratios
            ratio_mean_pl.append((mean_mc/mean_pl).cpu().numpy())
            ratio_var_pl.append((var_mc/var_pl).cpu().numpy())
            ratio_mean_mp.append((mean_mc/mean_mp).cpu().numpy())
            ratio_var_mp.append((var_mc/var_mp).cpu().numpy())

        print(model)
        print(f"MC runtime: {np.mean(time_mc):.8f} +- {np.std(time_mc):.8f}")
        print(f"PL-DNN runtime: {np.mean(time_pl):.8f} +- {np.std(time_pl):.8f}")
        print(f"Moment prop runtime: {np.mean(time_mp):.8f} +- {np.std(time_mp):.8f}")
        print(f"MC mean / PL-DNN mean: {np.mean(ratio_mean_pl):.4f} +- {np.std(ratio_mean_pl):.4f}")
        print(f"MC var / PL-DNN var: {np.mean(ratio_var_pl):.4f} +- {np.std(ratio_var_pl):.4f}")
        print(f"MC mean / MP mean: {np.mean(ratio_mean_mp):.4f} +- {np.std(ratio_mean_mp):.4f}")
        print(f"MC var / MP var: {np.mean(ratio_var_mp):.4f} +- {np.std(ratio_var_mp):.4f}")

    # Run convolutional experiments
    for network in [CNN_A, CNN_B]:
        torch.manual_seed(seed)
        time_mc = []
        time_pl = []
        time_mp = []
        ratio_mean_pl = []
        ratio_var_pl = []
        ratio_mean_mp = []
        ratio_var_mp = []

        input_size = 400
        input_shape = (20, 20)
        n_channels = 1
        
        for i in range(n_runs):
            # Set up input distribution
            mean = torch.randn(1,input_size, dtype=dtype, device=device)
            cov = nm.utils.rand.definite(input_size, dtype=dtype, device=device, 
                                        positive=True, semi=False, norm=1.0)
            input_dist = torch.distributions.MultivariateNormal(mean, cov)
        
            # Generate random MLP
            model = network(input_shape, n_channels)
            if dtype == torch.float64:
                model.double()
            model = model.to(device)
        
            # Monte Carlo estimate of output statistics
            start_t = time.time()
            samples = input_dist.sample((n_samples,))
            samples = samples.reshape(-1, n_channels, *input_shape)
            out_samples = model(samples).detach()
            mean_mc = torch.mean(out_samples, dim=0)
            var_mc = torch.var(out_samples, dim=0)
            end_t = time.time()
            time_mc.append(end_t - start_t)
        
            # Piecewise linear estimate of output statistics (Bibi, et al., 2018)
            start_t = time.time()
            x = mean.reshape(-1, n_channels, *input_shape)
            A, c1 = nm.utils.linearize(model.m1, x)
            B, c2 = nm.utils.linearize(model.m2, model.relu(model.m1(x)).detach())
            x.requires_grad_(False)
            A.squeeze_()
            c1.squeeze_()
            B.squeeze_()
            c2.squeeze_()
            mean_pl = gnm.mean(mean, cov, A, c1, B, c2)
            var_pl = gnm.special_variance(cov, A, B)
            end_t = time.time()
            time_pl.append(end_t - start_t)
        
            # Moment propagation of output statistics
            start_t = time.time()
            layers = [module for module in model.modules()]
            mean_mp, var_mp = propagate_moments(mean, cov, layers, input_shape=(n_channels, *input_shape))
            end_t = time.time()
            time_mp.append(end_t - start_t)
        
            # Compute ratios
            ratio_mean_pl.append((mean_mc/mean_pl).cpu().numpy())
            ratio_var_pl.append((var_mc/var_pl).cpu().numpy())
            ratio_mean_mp.append((mean_mc/mean_mp).cpu().numpy())
            ratio_var_mp.append((var_mc/var_mp).cpu().numpy())

        print(model)
        print(f"MC runtime: {np.mean(time_mc):.8f} +- {np.std(time_mc):.8f}")
        print(f"PL-DNN runtime: {np.mean(time_pl):.8f} +- {np.std(time_pl):.8f}")
        print(f"Moment prop runtime: {np.mean(time_mp):.8f} +- {np.std(time_mp):.8f}")
        print(f"MC mean / PL-DNN mean: {np.mean(ratio_mean_pl):.4f} +- {np.std(ratio_mean_pl):.4f}")
        print(f"MC var / PL-DNN var: {np.mean(ratio_var_pl):.4f} +- {np.std(ratio_var_pl):.4f}")
        print(f"MC mean / MP mean: {np.mean(ratio_mean_mp):.4f} +- {np.std(ratio_mean_mp):.4f}")
        print(f"MC var / MP var: {np.mean(ratio_var_mp):.4f} +- {np.std(ratio_var_mp):.4f}")