import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse, os, sys, json, logging

from datasets import generate_uci_data
from dvi.bayes_layers import BayesLinear, BayesReLU
from dvi.bayes_loss import RegressionLoss

class UCIDataset(Dataset):
    """UCI regression dataset"""
    def __init__(self, data, mean=None, std=None):
        if mean is None:
            self.mean = np.mean(data, axis=0)
        else:
            self.mean = mean
        if std is None:
            self.std = np.std(data, axis=0) + 1e-6
        else:
            self.std = std
        self.data = (data - self.mean)/self.std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index, :-1]   # Data
        y = self.data[index, -1]    # Label
        return x, y

class BNN(torch.nn.Module):
    """Bayesian neural network"""
    def __init__(self, args):
        super().__init__()
        input_size = args["input_size"]
        output_size = args["output_size"]
        layers = args["layers"]
        self.mode = args["mode"]
        if "prior_var" in args:
            self.prior_var = args["prior_var"]
        else:
            self.prior_var = None
        self.layers = torch.nn.Sequential()
        self.layers.append(BayesLinear(input_size, layers[0], 
                                       prior_var=self.prior_var, 
                                       mode=self.mode, 
                                       prior_type=args["prior_type"]))
        for i in range(1,len(layers)):
            self.layers.append(BayesReLU(layers[i-1], layers[i], 
                                         prior_var=self.prior_var, 
                                         mode=self.mode, 
                                         prior_type=args["prior_type"]))
        self.layers.append(BayesReLU(layers[-1], output_size, 
                                     prior_var=self.prior_var, 
                                     mode=self.mode, 
                                     prior_type=args["prior_type"]))
    def forward(self, x):
        out = self.layers(x)
        return out
    
def get_mse(pred, target):
    if type(pred) == tuple:
        pred_mean = pred[0][:,0]
    else:
        pred_mean = pred[:,0]
    return torch.nn.functional.mse_loss(pred_mean, target)

def train_epoch(model, train_loader, device, criterion, mean, std, optimizer):
    """Train BNN over a single epoch"""
    model.train()
    avg_elbo = avg_ll = avg_kl = rmse = 0.0
    for inputs, target in train_loader:
        optimizer.zero_grad()
        if criterion.mode == "mcvi":
            n_samples = 10 # Number of Monte Carlo samples per input
            inputs = torch.tile(inputs, (n_samples, 1))
            target = torch.tile(target, (n_samples,))
        inputs, target = inputs.to(device), target.to(device)
        pred = model(inputs)                        # Forward pass
        elbo, ll, kl, _ = criterion(pred, target)   # Compute loss
        elbo.backward()                             # Backpropagate
        if args["model"]["gradient_clip"] > 0.0:
            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), args["model"]["gradient_clip"])
        optimizer.step()                            # Update parameters
        mse = get_mse(pred, target)
        avg_elbo += elbo.item()
        avg_ll += ll.item()
        avg_kl += kl.item()
        rmse += np.sqrt(mse.item())*std
    avg_elbo /= len(train_loader)
    avg_ll /= len(train_loader)
    avg_kl /= len(train_loader)
    rmse /= len(train_loader)
    return avg_elbo, avg_ll, avg_kl, rmse

def validate_model(model, val_loader, device, criterion, mean, std):
    """Validate model"""
    avg_elbo = avg_ll = avg_kl = avg_xll = rmse = 0.0
    with torch.no_grad():
        model.eval()
        for inputs, target in val_loader:
            if criterion.mode == "mcvi":
                n_samples = 10
                inputs = torch.tile(inputs, (n_samples, 1))
                target = torch.tile(target, (n_samples,))
            inputs, target = inputs.to(device), target.to(device)
            pred = model(inputs)
            elbo, ll, kl, xll = criterion(pred, target, eval=True, n_samples=100)
            mse = get_mse(pred, target)
            avg_elbo += elbo.item()
            avg_ll += ll.item()
            avg_kl += kl.item()
            avg_xll += xll.item()
            rmse += np.sqrt(mse.item())*std
    avg_elbo /= len(val_loader) # Loss
    avg_ll /= len(val_loader)   # Log-likelihood (training metric)
    avg_kl /= len(val_loader)   # KL divergence
    avg_xll /= len(val_loader)  # Test log-likelihood for comparison
    rmse /= len(val_loader)     # Root mean squared error
    return avg_elbo, avg_ll, avg_kl, rmse, avg_xll

def log_experiment(run_id, xll_mean, xll_std, rmse_mean, rmse_std, ll_mean, ll_std, ):
    logging.info(
        f"{run_id}\tBest mean test log-likelihood: {xll_mean} +- {xll_std}\t"
        f"RMSE: {rmse_mean} +- {rmse_std}\t"
        f"DVI log-likelihood: {ll_mean} +- {ll_std}"
    )

if __name__ == "__main__":
    # Load configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Filepath for json config file, e.g. 'config/config.json'")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        verbose = True
    else:
        verbose = False
    with open(args.config) as f:
        args = json.load(f) # Set up args as a config dictionary
    mode = args["model"]["mode"]
    valid_modes = ["proposed", "dvi", "mcvi"]
    if mode not in valid_modes:
        print(f"Unsupported mode \"{mode}\" (Choose \"proposed\", \"dvi\", or \"mcvi\")")
    if not os.path.exists("out/"):
        os.makedirs("out/")
    logging.basicConfig(filename='out/experiments.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='a')

    # Assign random seed
    seed = args["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Check if cuda is available and assign device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 8 if cuda else 0
    print(f"cuda = {cuda} with num_workers = {num_workers}, system version = {sys.version}")

    # Generate data
    n_samples, input_size, data = generate_uci_data(args["dataset"])

    train_split = args["train_split"]
    args["model"]["input_size"] = input_size
    args["model"]["output_size"] = 2

    batch_size = args["model"]["batch_size"]
    train_args = {"shuffle": True,
                  "batch_size": batch_size}
    test_args = {"shuffle": False,
                 "batch_size": batch_size}

    # Run experiments
    if "n_epochs" in args["model"]:
        n_epochs = args["model"]["n_epochs"]
    else:
        n_epochs = max(int(500000/batch_size/(args["train_split"]*len(data))), 1)
    train_ll_all = []
    train_rmse_all = []
    test_ll_all = []
    test_rmse_all = []
    test_xll_all = []
    for i in tqdm(range(args["n_runs"])):
        if verbose: print(f"========== Run iteration {i+1}... ==========")

        # Set up train/test split
        data_shuffled = data
        np.random.shuffle(data_shuffled)
        train_data, test_data = np.split(data_shuffled, [int(train_split*n_samples)])
        train_data = UCIDataset(train_data)
        out_mean = train_data.mean[-1]
        out_std = train_data.std[-1]
        out_log_std = np.log(out_std)
        test_data = UCIDataset(test_data, train_data.mean, train_data.std)
        train_loader = DataLoader(train_data, **train_args)
        test_loader = DataLoader(test_data, **test_args)

        # Define model
        model = BNN(args["model"])
        model = model.to(device)
        loss = RegressionLoss(model, args["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["model"]["learning_rate"], weight_decay=0.001)
        scheduler = None

        # Train
        train_ll = []
        train_rmse = []
        test_ll = []
        test_xll = []
        test_rmse = []
        for epoch in range(n_epochs):
            elbo, ll, kl, rmse = train_epoch(model, train_loader, device, loss, out_mean, out_std, optimizer)
            train_ll.append(ll - out_log_std)
            train_rmse.append(rmse)
            if verbose:
                print(f"Epoch: {epoch+1}")
                print(f"Train\tLoss: {elbo:.4f}", end="")
                print(f"\tLL: {ll - out_log_std:.4f}", end="")
                print(f"\tKL: {kl:.4f}", end="")
                print(f"\tRMSE: {rmse:.4f}")
            elbo, ll, kl, rmse, xll = validate_model(model, test_loader, device, loss, out_mean, out_std)
            test_ll.append(ll - out_log_std)
            test_rmse.append(rmse)
            test_xll.append(xll - out_log_std)
            if verbose:
                print(f"Test\tLoss: {elbo:.4f}", end="")
                print(f"\tLL: {ll - out_log_std:.4f}", end="")
                print(f"\tKL: {kl:.4f}", end="")
                print(f"\tRMSE: {rmse:.4f}")
                print(f"\txLL: {xll - out_log_std:.4f}")
                print("="*40)
            if scheduler:
                scheduler.step()
        train_ll = np.array(train_ll)
        train_rmse = np.array(train_rmse)
        test_ll = np.array(test_ll)
        test_rmse = np.array(test_rmse)
        test_xll = np.array(test_xll)
        train_ll_all.append(train_ll)
        train_rmse_all.append(train_rmse)
        test_ll_all.append(test_ll)
        test_rmse_all.append(test_rmse)
        test_xll_all.append(test_xll)
        if verbose:
            x = np.arange(1, n_epochs+1)
            plt.plot(x, train_ll, 'b', label="Train")
            plt.plot(x, test_ll, 'r', label="Test")
            plt.xlabel('Epochs')
            plt.ylabel('Log-Likelihood')
            plt.legend()
            plt.grid(True)
            plt.show()

    # Find the best test log-likelihood and return metrics from that index
    train_ll_all = np.vstack(train_ll_all)
    train_rmse_all = np.vstack(train_rmse_all)
    test_ll_all = np.vstack(test_ll_all)
    test_rmse_all = np.vstack(test_rmse_all)
    test_xll_all = np.vstack(test_xll_all)
    idx = np.argmax(test_xll_all, axis=1)
    test_xll_best = test_xll_all[np.arange(args["n_runs"]),idx]
    test_xll_mean = np.mean(test_xll_best)
    test_xll_std = np.std(test_xll_best)
    test_ll_best = test_ll_all[np.arange(args["n_runs"]),idx]
    test_ll_mean = np.mean(test_ll_best)
    test_ll_std = np.std(test_ll_best)
    test_rmse_best = test_rmse_all[np.arange(args["n_runs"]),idx]
    test_rmse_mean = np.mean(test_rmse_best)
    test_rmse_std = np.std(test_rmse_best)
    print(f"Best test log-likelihood: {test_xll_mean} +- {test_xll_std}\tRMSE: {test_rmse_mean} +- {test_rmse_std}")
    run_id = "{}_{}_{}_{}_{}_{}_{}_{}".format(args["dataset"], 
                                        args["model"]["mode"],
                                        args["n_runs"],
                                        n_epochs,
                                        args["model"]["learning_rate"],
                                        args["model"]["lambda"],
                                        args["model"]["prior_type"][1],
                                        args["model"]["prior_var"])
    log_experiment(run_id, test_xll_mean, test_xll_std, test_rmse_mean, test_rmse_std, test_ll_mean, test_ll_std)  