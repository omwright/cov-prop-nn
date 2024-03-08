import torch
import torch.nn as nn
import numpy as np

from dvi.bayes_layers import BayesLinear, BayesReLU
import dvi.bayes_utils as bu

class RegressionLoss(nn.Module):
    def __init__(self, model, args):
        """
        ELBO loss, a surrogate objective for the KL divergence 
        between the parameterized distribution and the true posterior
        """
        super().__init__()
        self.model = model
        self.mode = args["mode"]
        self.warmup = args["warmup_updates"]
        self.anneal = args["anneal_updates"]
        self.batch_size = args["batch_size"]
        self.lmbda = args["lambda"]
    
    def _bayesian_kl_loss(self):
        kl = 0.0
        for m in self.model.modules():
            if isinstance(m, (BayesLinear, BayesReLU)):
                kl += bu.kl_loss(m.weight_mean, m.weight_logvar, m.prior_mean, m.prior_logvar) + \
                    bu.kl_loss(m.bias_mean, m.bias_logvar, m.prior_mean, m.prior_logvar)
        return kl

    def _gaussian_loglikelihood_core(self, target, mean, log_var, smm, sml, sll):
        """Reconstruction term for Gaussian with heteroskedastic variance"""
        return -0.5*(bu.log2pi + log_var + \
                     torch.exp(-log_var + 0.5*sll)*(smm + (mean - sml - target)**2))   

    def _heteroskedastic_gaussian_loglikelihood(self, pred, target, eval=False):
        if type(pred) is tuple:
            pred_mean, pred_var = pred
        else:
            pred_mean = pred
        mean = pred_mean[:,0].reshape(-1)       # Mean of mean variable
        log_var = pred_mean[:,1].reshape(-1)    # Mean of logvar variable
        if self.mode == "mcvi" or eval is True:
            sll = smm = sml = 0.0
        elif self.mode == "proposed" or self.mode == "dvi":
            smm = pred_var[:, 0, 0].reshape(-1)     # Variance of mean
            sll = pred_var[:, 1, 1].reshape(-1)     # Variance of logvar
            sml = pred_var[:, 0, 1].reshape(-1)     # Covariance between mean and logvar
        return self._gaussian_loglikelihood_core(target, mean, log_var, smm, sml, sll)
            
    def forward(self, pred, target, step=None, eval=False, n_samples=10):
        kl = self._bayesian_kl_loss()
        if self.mode == "mcvi":
            batch_size = self.batch_size*n_samples
        else:
            batch_size = self.batch_size
        surprise = kl/batch_size
        
        # Note: Only heteroskedastic is implemented
        log_likelihood = self._heteroskedastic_gaussian_loglikelihood(pred, target)
        batch_log_likelihood = torch.mean(log_likelihood)
        if eval is True: # For apples-to-apples figure of merit
            if self.mode == "mcvi":
                x_log_likelihood = batch_log_likelihood
            else:
                x_log_likelihood = self._heteroskedastic_gaussian_loglikelihood(pred, target, eval=True)
                x_log_likelihood = torch.mean(x_log_likelihood)
        else:
            x_log_likelihood = None
        if step:
            lmbda = np.clip((step - self.warmup)/self.anneal, 0.0, 1.0)
        else:
            lmbda = self.lmbda
        loss = lmbda*surprise - batch_log_likelihood
        return loss, batch_log_likelihood, surprise, x_log_likelihood