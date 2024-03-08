import torch
import numpy as np

EPSILON = 1e-6

pi = np.pi
sqrt2 = np.sqrt(2.0)
sqrt2pi = np.sqrt(2.0)*pi
one_ovr_sqrt2 = 1.0/sqrt2
one_ovr_sqrt2pi = 1.0/sqrt2pi
log2pi = np.log(2*pi)

def standard_gaussian(x):
    return one_ovr_sqrt2pi * torch.exp(-x*x / 2.0)

def gaussian_cdf(x):
    return 0.5 * (1.0 + torch.erf(x * one_ovr_sqrt2))

def softrelu(x):
    return standard_gaussian(x) + x*gaussian_cdf(x)

def q(rho, mu1, mu2):
    """Return the ReLU error-correction term"""
    one_plus_rho_hat = torch.sqrt(1 - rho*rho) + 1.0
    alpha = torch.asin(rho) - rho/one_plus_rho_hat
    safe_alpha = torch.abs(alpha) + EPSILON/2.0
    safe_rho = torch.abs(rho) + EPSILON
    beta = safe_rho/(2.0*safe_alpha*one_plus_rho_hat)
    gamma = (rho - torch.asin(rho))/(safe_rho*safe_alpha)

    return (alpha/(2*pi)) * torch.exp(-(mu1*mu1 + mu2*mu2)*beta - gamma*mu1*mu2)

def delta(rho, mu1, mu2):
    return gaussian_cdf(mu1) * gaussian_cdf(mu2) + q(rho, mu1, mu2)

def linear_covariance(x_mean, x_cov, weights_mean, weights_var, bias_mean, bias_var):
    """Return covariance of a linear transformation"""
    # Compute the three terms of Eq. 3 in (Wu, et al., 2019)
    x_var_diag = torch.diagonal(x_cov, dim1=-1, dim2=-2)
    xx_mean = x_var_diag + x_mean*x_mean # Second moment of input
    term1_diag = xx_mean@weights_var

    flat_xCov = torch.reshape(x_cov, (-1, weights_mean.shape[0]))
    xCov_A = flat_xCov@weights_mean
    xCov_A = torch.reshape(xCov_A, (-1, weights_mean.shape[0], weights_mean.shape[1]))
    xCov_A = torch.transpose(xCov_A, 1, 2) # (b,y,y)
    xCov_A = torch.reshape(xCov_A, (-1, weights_mean.shape[0]))
    A_xCov_A = xCov_A@weights_mean
    A_xCov_A = torch.reshape(A_xCov_A, (-1, weights_mean.shape[1], weights_mean.shape[1]))
    term2 = A_xCov_A
    term2_diag = torch.diagonal(term2, dim1=-1, dim2=-2)

    term3_diag = bias_var

    result_diag = term1_diag + term2_diag + term3_diag
    idx = torch.arange(0, term2.shape[1])
    result = term2
    result[:,idx,idx] = result_diag
    return result 

def relu_covariance(x_var, x_var_diag, mu):
    """Return covariance of a ReLU activation using (Wu, et al., 2019)"""
    mu1 = torch.unsqueeze(mu, 2)
    mu2 = torch.transpose(mu1, 1, 2)
    s11s22 = torch.unsqueeze(x_var_diag, dim=2)*torch.unsqueeze(x_var_diag, dim=1)
    rho = x_var/(torch.sqrt(s11s22))
    rho = torch.clamp(rho, -1/(1+EPSILON), 1/(1+EPSILON))
    return x_var*delta(rho, mu1, mu2)

def relu_covariance_new(x_mean, x_cov, x_var_diag, x_std_diag, m):
    """Return covariance of a ReLU activation using proposed method
    """
    m1 = torch.unsqueeze(m, 2)      # Unitless mean/std variable
    m2 = torch.transpose(m1, 1, 2)
    x_mean1 = torch.unsqueeze(x_mean, 2)
    x_mean2 = torch.transpose(x_mean1, 1, 2)
    x_std_diag1 = torch.unsqueeze(x_std_diag, 2)
    x_std_diag2 = torch.transpose(x_std_diag1, 1, 2)
    s11s22 = torch.unsqueeze(x_var_diag, dim=2)*torch.unsqueeze(x_var_diag, dim=1)
    rho = x_cov/(torch.sqrt(s11s22)) # Correlation coefficient
    rho = torch.clamp(rho, -1/(1+EPSILON), 1/(1+EPSILON))

    t1 = rho * x_std_diag1 * gaussian_cdf(m1) * x_std_diag2 * gaussian_cdf(m2)
    t2 = (1./2) * rho**2 * x_std_diag1 * standard_gaussian(m1) * x_std_diag2 * standard_gaussian(m2)
    t3 = (1./6) * rho**3 * -x_mean1 * standard_gaussian(m1) * -x_mean2 * standard_gaussian(m2)
    t4 = (1./24) * rho**4 * x_std_diag1*(m1**2 - 1) * standard_gaussian(m1) * x_std_diag2*(m2**2 - 1) * standard_gaussian(m2)
    t5 = (1./120) * rho**5 * -x_mean1*(m1**3 - 3) * standard_gaussian(m1) * -x_mean2*(m2**3 - 3) * standard_gaussian(m2)

    return t1 + t2 + t3 + t4 + t5

def kl_loss(p_mean, p_logvar, q_mean, q_logvar):
    """Return KL divergence between two Gaussian distributions"""
    kl = 0.5*(q_logvar - p_logvar - 1.0 + (torch.exp(p_logvar) + (p_mean - q_mean)**2)/(np.exp(q_logvar) + EPSILON))
    return torch.sum(kl)