import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import dvi.bayes_utils as bu

class BayesLinear(nn.Module):
    """Applies a linear transformation y = xW^T + b
    where parameters have a Gaussian prior.

    Args:
        in_features:    Input dimension
        out_features:   Output dimension
        prior_mean:     Mean of prior Gaussian distribution
        prior_var:      Covariance of prior Gaussian distribution (use prior_type if None)
        bias:           Learn additive bias if True
        mode:           Variational inference method ("mcvi")
        prior_type[0]:  Type of mean prior
        prior_type[1]:  Type of variance prior
    """

    def __init__(self, in_features, out_features, prior_mean=0.0, prior_var=None, mode="dvi", prior_type=["manual", "wider_he"]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.weight_mean = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias_mean = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        self.mode = mode
        self.prior_type = prior_type
        self.reset_parameters()

    def reset_parameters(self):
        # Set variance scale, following (Wu, et al., 2019)
        if self.prior_var and self.prior_type[1] == "manual":
            scale = self.prior_var
        elif self.prior_type[1] == "he":
            scale = 2.0/self.in_features
        elif self.prior_type[1] == "wider_he":
            scale = 5.0/self.in_features
        elif self.prior_type[1] == "xavier":
            scale = 2.0/(self.in_features + self.out_features)
        self.prior_logvar = math.log(scale)

        # Initialize weights and biases
        nn.init.normal_(self.weight_mean, self.prior_mean, math.sqrt(scale))
        nn.init.normal_(self.bias_mean, self.prior_mean, math.sqrt(scale/100))
        self.weight_logvar.data.fill_(self.prior_logvar)
        self.bias_logvar.data.fill_(self.prior_logvar - 4.60517)

    def forward(self, x):
        """Propagate mean and covariance"""
        if self.mode == "dvi" or self.mode == "proposed":
            weight_var = torch.exp(self.weight_logvar)
            bias_var = torch.exp(self.bias_logvar)
            if type(x) is tuple:
                x_mean, x_cov = x
                y_cov = bu.linear_covariance(x_mean, x_cov, 
                                            self.weight_mean, weight_var, 
                                            self.bias_mean, bias_var)
            else: # Input samples instead of (mean, var) tuple
                x_mean = x
                xx = x_mean * x_mean
                y_cov = torch.diag_embed(xx@weight_var + bias_var)
            y_mean = x_mean@self.weight_mean + self.bias_mean
            return y_mean, y_cov
        elif self.mode == "mcvi":
            # There are lots of ways to set up MCVI; here we follow (Wu, et al., 2019)
            weight_var = torch.exp(self.weight_logvar)
            bias_var = torch.exp(self.bias_logvar)
            if type(x) is tuple:
               x_mean, x_cov = x
               y_cov = bu.linear_covariance(x_mean, x_cov, 
                                            self.weight_mean, weight_var, 
                                            self.bias_mean, bias_var)
            else:
                x_mean = x
                xx = x_mean * x_mean
                y_cov = torch.diag_embed(xx@weight_var + bias_var)
            y_mean = x_mean@self.weight_mean + self.bias_mean
            d = torch.distributions.MultivariateNormal(y_mean, y_cov)
            return d.rsample()

class BayesReLU(BayesLinear):
    """Applies the non-linear activation y = ReLU(x)W^T + b
    
    Note: Following the conventions of (Wu et al., 2019), the activation
    is applied in an atypical way.
    """
    def relu(self, x):
        if type(x) is tuple:
            x_mean, x_cov = x
        else:
            x_mean = x
            x_cov = None
        if x_cov is None:   
            z_mean = F.relu(x_mean)
            z_cov = None 
        else:
            x_var_diag = torch.diagonal(x_cov, dim1=-1, dim2=-2)
            sqrt_x_var_diag = torch.sqrt(x_var_diag)    # Standard deviation
            mu = x_mean/(sqrt_x_var_diag + bu.EPSILON)  # Dimensionless variable used by (Wu, et al., 2019)
            z_mean = sqrt_x_var_diag*bu.softrelu(mu)
            if self.mode == "dvi":
                z_cov = bu.relu_covariance(x_cov, x_var_diag, mu)
            elif self.mode == "proposed":
                z_cov = bu.relu_covariance_new(x_mean, x_cov, x_var_diag, sqrt_x_var_diag, mu)
        return z_mean, z_cov

    def forward(self, x):
        """Propagate expectation and covariance
        Args:
            x: input (batch, in_features)

        Returns:
            y_mean: (batch, out_features)
            y_cov: (batch, out_features, out_features)
        """
        if self.mode == "dvi" or self.mode == "proposed":
            x = self.relu(x)
            weight_var = torch.exp(self.weight_logvar)
            bias_var = torch.exp(self.bias_logvar)
            if type(x) is tuple:
                x_mean, x_cov = x
                y_cov = bu.linear_covariance(x_mean, x_cov, 
                                            self.weight_mean, weight_var, 
                                            self.bias_mean, bias_var)
            else:
                x_mean = x
                xx = x_mean * x_mean
                y_cov = torch.diag_embed(xx@weight_var + bias_var)
            y_mean = x_mean@self.weight_mean + self.bias_mean
            return y_mean, y_cov
        elif self.mode == "mcvi":
            x = self.relu(x)
            weight_var = torch.exp(self.weight_logvar)
            bias_var = torch.exp(self.bias_logvar)
            if type(x) is tuple:
                x_mean, x_cov = x
                if x_cov is None:
                    xx = x_mean * x_mean
                    y_cov = torch.diag_embed(xx@weight_var + bias_var)
                else:
                    y_cov = bu.linear_covariance(x_mean, x_cov, 
                                                self.weight_mean, weight_var, 
                                                self.bias_mean, bias_var)
            else:
                x_mean = x
                xx = x_mean * x_mean
                y_cov = torch.diag_embed(xx@weight_var + bias_var)
            y_mean = x_mean@self.weight_mean + self.bias_mean

            d = torch.distributions.MultivariateNormal(y_mean, y_cov)
            return d.rsample()