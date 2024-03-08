import os, sys, time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import network_moments.torch as nm
gnm = nm.gaussian.affine_relu_affine
from train_mnist import CNet

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
    """Propagate moments through nn.ReLU"""
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

if __name__ == "__main__":
    # Configure experiment
    n_samples = int(7.5e4)
    n_runs = 200
    fpath = "./data/"
    dtype = torch.float32
    seed = 42

    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 8 if cuda else 0
    print(f"Cuda = {cuda} with num_workers = {num_workers}, system version = {sys.version}")

    # Load trained network
    model = CNet(1, 10)
    model.load_state_dict(torch.load("./data/cnet.pt"))
    if dtype == torch.float64:
        model.double()
    model = model.to(device)
    model.eval()

    n_channels = 1
    input_shape = (28, 28)
    input_size = 28*28

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    test_set = datasets.MNIST(root=fpath, 
                            train=False, 
                            download=True,
                            transform=transform)

    time_mc = []
    time_pl = []
    time_mp = []
    ratio_mean_pl = []
    ratio_var_pl = []
    ratio_cov_pl = []
    ratio_mean_mp = []
    ratio_var_mp = []
    ratio_cov_mp = []
    for i in tqdm(range(n_runs)):
        # Set up input distribution
        idx = np.random.randint(0, len(test_set)) # Randomly select test image for mean
        mean, _ = test_set[idx]
        if dtype == torch.float64:
            mean.double()
        cov = nm.utils.rand.definite(input_size, dtype=dtype, device=device, 
                                    positive=True, semi=False, norm=1.0)
        input_dist = torch.distributions.MultivariateNormal(mean.reshape(1, input_size), cov)
        
        # Monte Carlo estimate of output statistics
        start_t = time.time()
        samples = input_dist.sample((n_samples,))
        samples = samples.reshape(-1, n_channels, *input_shape)
        # Using DataLoader for memory management
        sample_loader = torch.utils.data.DataLoader(samples, 
                                                    batch_size=1000, 
                                                    shuffle=False, 
                                                    num_workers=num_workers)
        out_list = []
        with torch.no_grad():
            for batch_idx, subsample in enumerate(sample_loader):
                out_subsample = model(subsample).detach()
                out_list.append(out_subsample)
        out_samples = torch.cat(out_list, dim=0)
        mean_mc = torch.mean(out_samples, dim=0)
        var_mc = torch.var(out_samples, dim=0)
        cov_mc = torch.cov(out_samples.T)
        end_t = time.time()
        time_mc.append(end_t - start_t)

        # Piecewise linear estimate of output statistics (Bibi, et al., 2018)
        start_t = time.time()
        x = mean.reshape(-1, n_channels, *input_shape) # Linearization point
        A, c1 = nm.utils.linearize(model.features, x)
        B, c2 = nm.utils.linearize(model.classifier, model.relu(model.features(x)).detach())
        x.requires_grad_(False)
        A.squeeze_()
        c1.squeeze_()
        B.squeeze_()
        c2.squeeze_()
        mean_pl = gnm.mean(mean.reshape(-1, input_size), cov, A, c1, B, c2)
        var_pl = gnm.special_variance(cov, A, B)
        cov_pl = gnm.special_covariance(cov, A, B)
        end_t = time.time()
        time_pl.append(end_t - start_t)

        # Moment propagation of output statistics
        start_t = time.time()
        layers = [module for module in model.modules()]
        mean_mp, cov_mp = propagate_moments(mean, cov, layers, input_shape=(n_channels, *input_shape))
        var_mp = torch.diag(cov_mp)
        end_t = time.time()
        time_mp.append(end_t - start_t)

        # Compute ratios
        ratio_mean_pl.append((mean_mc/mean_pl).cpu().numpy())
        ratio_var_pl.append((var_mc/var_pl).cpu().numpy())
        ratio_cov_pl.append((cov_mc/cov_pl).cpu().numpy())
        ratio_mean_mp.append((mean_mc/mean_mp).cpu().numpy())
        ratio_var_mp.append((var_mc/var_mp).cpu().numpy())
        ratio_cov_mp.append((cov_mc/cov_mp).cpu().numpy())

    print(model)
    fpath = "./out/"
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    np.savetxt(fpath+"mean_ratio_cov_pl.csv", np.mean(ratio_cov_pl, axis=0), delimiter=", ")
    np.savetxt(fpath+"std_ratio_cov_pl.csv", np.std(ratio_cov_pl, axis=0), delimiter=", ")
    np.savetxt(fpath+"mean_ratio_cov_mp.csv", np.mean(ratio_cov_mp, axis=0), delimiter=", ")
    np.savetxt(fpath+"std_ratio_cov_mp.csv", np.std(ratio_cov_mp, axis=0), delimiter=", ")
    print(f"MC runtime: {np.mean(time_mc):.8f} +- {np.std(time_mc):.8f}")
    print(f"PL-DNN runtime: {np.mean(time_pl):.8f} +- {np.std(time_pl):.8f}")
    print(f"Moment prop runtime: {np.mean(time_mp):.8f} +- {np.std(time_mp):.8f}")
    print(f"MC mean / PL-DNN mean: {np.mean(ratio_mean_pl, axis=0)} +- {np.std(ratio_mean_pl, axis=0)}")
    print(f"MC var / PL-DNN var: {np.mean(ratio_var_pl, axis=0)} +- {np.std(ratio_var_pl, axis=0)}")
    print(f"MC mean / MP mean: {np.mean(ratio_mean_mp, axis=0)} +- {np.std(ratio_mean_mp, axis=0)}")
    print(f"MC var / MP var: {np.mean(ratio_var_mp, axis=0)} +- {np.std(ratio_var_mp, axis=0)}")