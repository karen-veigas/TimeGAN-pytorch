"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen(robinbg@foxmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .data import batch_generator
from utils import extract_time, random_generator, NormMinMax
from .model import Encoder, Recovery, Generator, Discriminator, Supervisor

import matplotlib.pyplot as plt
import numpy as np


class BaseModel():
  """ Base Model for timegan
  """
  loss_history = {
        "er": [],  # Reconstruction loss (Encoder/Recovery)
        "s": [],   # Supervision loss (Supervisor)
        "g": [],   # Generator loss
        "d": []    # Discriminator loss
    }

  def __init__(self, opt, ori_data):
    # Seed for deterministic behavior
    self.seed(opt.manualseed)

    # Initalize variables.
    self.opt = opt
    self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
    self.ori_time, self.max_seq_len = extract_time(self.ori_data)
    self.data_num, _, _ = np.asarray(ori_data).shape    # 3661; 24; 6
    self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
    self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
    self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

  def seed(self, seed_value):
    """ Seed

    Arguments:
        seed_value {int} -- [description]
    """
    # Check if seed is default value
    if seed_value == -1:
      return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def save_weights(self, epoch):
    """Save net weights for the current epoch.

    Args:
        epoch ([int]): Current epoch number.
    """

    weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
    if not os.path.exists(weight_dir): 
      os.makedirs(weight_dir)

    torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
               '%s/netE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
               '%s/netR.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
               '%s/netG.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
               '%s/netD.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
               '%s/netS.pth' % (weight_dir))


  def train_one_iter_er(self):
    """ Train the model for one epoch.
    """

    self.nete.train()
    self.netr.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
   # self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.X = torch.from_numpy(np.asarray(self.X0, dtype=np.float32)).to(self.device)

    cond = np.ones((self.X.shape[0], self.X.shape[1], self.opt.cond_dim), dtype=np.float32)
    self.C = torch.tensor(cond, device=self.device)

    # train encoder & decoder
    self.optimize_params_er()

  def train_one_iter_er_(self):
    """ Train the model for one epoch.
    """

    self.nete.train()
    self.netr.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
   # self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.X = torch.from_numpy(np.asarray(self.X0, dtype=np.float32)).to(self.device)

    # Simple placeholder condition for now (expand later)
    cond = np.ones((self.X.shape[0], self.X.shape[1], self.opt.cond_dim), dtype=np.float32)
    self.C = torch.tensor(cond, device=self.device)

    # train encoder & decoder
    self.optimize_params_er_()
 
  def train_one_iter_s(self):
    """ Train the model for one epoch.
    """

    #self.nete.eval()
    self.nets.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
   # self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.X = torch.from_numpy(np.asarray(self.X0, dtype=np.float32)).to(self.device)

    # Simple placeholder condition for now (expand later)
    cond = np.ones((self.X.shape[0], self.X.shape[1], self.opt.cond_dim), dtype=np.float32)
    self.C = torch.tensor(cond, device=self.device)

    
    # train superviser
    self.optimize_params_s()

  def train_one_iter_g(self):
    """ Train the model for one epoch.
    """

    """self.netr.eval()
    self.nets.eval()
    self.netd.eval()"""
    self.netg.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
   # self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.X = torch.from_numpy(np.asarray(self.X0, dtype=np.float32)).to(self.device)

    # Simple placeholder condition for now (expand later)
    cond = np.ones((self.X.shape[0], self.X.shape[1], self.opt.cond_dim), dtype=np.float32)
    self.C = torch.tensor(cond, device=self.device)

    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_g()

  def train_one_iter_d(self):
    """ Train the model for one epoch.
    """
    """self.nete.eval()
    self.netr.eval()
    self.nets.eval()
    self.netg.eval()"""
    self.netd.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
   # self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.X = torch.from_numpy(np.asarray(self.X0, dtype=np.float32)).to(self.device)

    # Simple placeholder condition for now (expand later)
    cond = np.ones((self.X.shape[0], self.X.shape[1], self.opt.cond_dim), dtype=np.float32)
    self.C = torch.tensor(cond, device=self.device)

    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_d()


  def train(self):
    """ Train the model
    """

    for iter in range(self.opt.iteration):
      # Train for one iter
      self.train_one_iter_er()

      print('Encoder training step: '+ str(iter) + '/' + str(self.opt.iteration))

    for iter in range(self.opt.iteration):
      # Train for one iter
      self.train_one_iter_s()

      print('Superviser training step: '+ str(iter) + '/' + str(self.opt.iteration))

    for iter in range(self.opt.iteration):
      # for _ in range(self.opt.n_critic):
      #     self.train_one_iter_d()      # critic with GP
      self.train_one_iter_g()          # generator (includes supervisor forward)
      self.train_one_iter_er_()        # small reconstruction refresh
      
      # Early Stopping
      if len(BaseModel.loss_history["d"]) > 50:
        recent_d = np.mean(BaseModel.loss_history["d"][-50:])
        if recent_d < 0.01:
            print("⚠ Early stopping: Discriminator collapsed")
            break

      print('Superviser training step: '+ str(iter) + '/' + str(self.opt.iteration))

    self.save_weights(self.opt.iteration)
    self.generated_data = self.generation(self.opt.batch_size)
    print('Finish Synthetic Data Generation')
    self.plot_losses(BaseModel.loss_history)

  #  self.evaluation()
  @staticmethod
  def smooth(values, alpha=0.95):
      """ Exponential smoothing to reduce noise """
      smoothed = []
      v = 0
      for x in values:
          v = alpha * v + (1 - alpha) * x
          smoothed.append(v)
      return smoothed

  @staticmethod
  def plot_losses(loss_history, smooth_curves=True):
      plt.figure(figsize=(14, 6))

      for key, values in loss_history.items():
          if len(values) == 0:
              continue  # skip empty series

          if smooth_curves:
              plt.plot(BaseModel.smooth(values), label=f"{key.upper()} (smoothed)", alpha=0.9)
          else:
              plt.plot(values, label=key.upper(), alpha=0.7)

      plt.title("✅ TimeGAN Training Losses", fontsize=16)
      plt.xlabel("Training steps", fontsize=12)
      plt.ylabel("Loss value", fontsize=12)
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()
      
      
  """def evaluation(self):
    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(self.opt.metric_iteration):
      temp_disc = discriminative_score_metrics(self.ori_data, self.generated_data)
      discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(self.opt.metric_iteration):
      temp_pred = predictive_score_metrics(self.ori_data, self.generated_data)
      predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(self.ori_data, self.generated_data, 'pca')
    visualization(self.ori_data, self.generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)
"""

  def generation_old(self, num_samples, mean = 0.0, std = 1.0):
    if num_samples == 0:
      return None, None
    ## Synthetic data generation
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.Z = random_generator(num_samples, self.opt.z_dim, self.T, self.max_seq_len)
    self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
    self.E_hat = self.netg(self.Z)    # [?, 24, 24]
    self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]
    generated_data_curr = self.netr(self.H_hat).cpu().detach().numpy()  # [?, 24, 24]

    generated_data = list()
    for i in range(num_samples):
      temp = generated_data_curr[i, :self.ori_time[i], :]
      generated_data.append(temp)
    
    # Renormalization
    generated_data = generated_data * self.max_val
    generated_data = generated_data + self.min_val
    return generated_data
  
  def generation(self, num_samples, regime=0):
      # Z as before
      T = np.array([self.max_seq_len]*num_samples)
      Z = random_generator(num_samples, self.opt.z_dim, T, self.max_seq_len)
      Z = torch.tensor(Z, dtype=torch.float32, device=self.device)

      # Build condition sequence
      cond = np.eye(self.opt.cond_dim, dtype=np.float32)[regime]  # regime int or array
      if isinstance(regime, (int, np.integer)):
          C = np.repeat(cond[None,None,:], num_samples*self.max_seq_len, axis=0)
      else:
          # per-sample regimes
          C = np.eye(self.opt.cond_dim, dtype=np.float32)[np.array(regime)]
          C = np.repeat(C[:,None,:], self.max_seq_len, axis=1)
      C = torch.tensor(C, dtype=torch.float32, device=self.device)

      with torch.no_grad():
          E_hat = self.netg(Z)
          H_hat = self.nets(E_hat)
          X_hat = self.netr(H_hat).cpu().numpy()

      return X_hat.astype(np.float32)  # still in [0,1] if you trained on scaled windows





class TimeGAN(BaseModel):
    """TimeGAN Class
    """

    @property
    def name(self):
      return 'TimeGAN'

    def __init__(self, opt, ori_data):
      super(TimeGAN, self).__init__(opt, ori_data)

      # -- Misc attributes
      self.epoch = 0
      self.times = []
      self.total_steps = 0

      # Create and initialize networks.
      self.nete = Encoder(self.opt).to(self.device)
      self.netr = Recovery(self.opt).to(self.device)
      self.netg = Generator(self.opt).to(self.device)
      self.netd = Discriminator(self.opt).to(self.device)
      self.nets = Supervisor(self.opt).to(self.device)

      if self.opt.resume != '':
        print("\nLoading pre-trained networks.")
        self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])
        print("\tDone.\n")

      # loss
      self.l_mse = nn.MSELoss()
      self.l_r = nn.L1Loss()
      # self.l_bce = nn.BCELoss()

      # Setup optimizer
      if self.opt.isTrain:
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()
        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        # self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(),lr=self.opt.lr * 0.2, betas=(self.opt.beta1, 0.999))
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def wasserstein_loss(self, y_pred, y_true):
        return torch.mean(y_true * y_pred)
    
    
    def grad_penalty(self, real, fake, cond):
        # real,fake: [B,T,Hdim] ; cond: [B,T,cond_dim]
        alpha = torch.rand(real.size(0), 1, 1, device=real.device)
        inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        inter_c = torch.cat([inter, cond], dim=-1)
        out = self.netd(inter_c)

        grad = torch.autograd.grad(outputs=out,inputs=inter,grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True,only_inputs=True)[0]

        gp = ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1.0)**2).mean()
        return gp


    def forward_e(self):
      """ Forward propagate through netE
      """
      Xc = torch.cat([self.X, self.C], dim=-1)
      self.H = self.nete(Xc)

    def forward_er(self):
      """ Forward propagate through netR
      """
      Xc = torch.cat([self.X, self.C], dim=-1)
      self.H = self.nete(Xc)
      self.X_tilde = self.netr(self.H)

    def forward_g(self):
      """ Forward propagate through netG
      """
      self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
      self.E_hat = self.netg(self.Z)
    def forward_dg(self):
      """ Forward propagate through netD
        """
      H_hatc = torch.cat([self.H_hat, self.C], dim=-1)
      self.Y_fake = self.netd(H_hatc)
      self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
      """ Forward propagate through netG
      """
      self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
      """ Forward propagate through netS
      """
      self.H_supervise = self.nets(self.H)
      # print(self.H, self.H_supervise)

    def forward_sg(self):
      """ Forward propagate through netS
      """
      self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
      """ Forward propagate through netD
      """
      
      Hc = torch.cat([self.H, self.C], dim=-1)
      H_hatc = torch.cat([self.H_hat, self.C], dim=-1)

      self.Y_real = self.netd(Hc)
      self.Y_fake = self.netd(H_hatc)
      self.Y_fake_e = self.netd(self.E_hat)


    def backward_er(self):
      """ Backpropagate through netE
      """
      self.err_er = self.l_mse(self.X_tilde, self.X)
      self.err_er.backward(retain_graph=True)
      BaseModel.loss_history["er"].append(self.err_er.item())
      print("Loss: ", self.err_er)

    def backward_er_(self):
      """ Backpropagate through netE
      """
      self.err_er_ = self.l_mse(self.X_tilde, self.X) 
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
      self.err_er.backward(retain_graph=True)
      BaseModel.loss_history["er"].append(self.err_er.item())


    def backward_g(self):
        # ==========================
        #   WGAN Generator Loss
        # ==========================

        # Conditioned fake latent
        H_fake_c = torch.cat([self.H_hat, self.C], dim=-1)
        G_adv = -self.netd(H_fake_c).mean()   # minimize -D(fake)

        # Moment matching losses (unchanged from your version)
        V1 = torch.mean(torch.abs(torch.sqrt(torch.var(self.X_hat,[0])[1] + 1e-6)
                                - torch.sqrt(torch.var(self.X,[0])[1] + 1e-6)))
        V2 = torch.mean(torch.abs(torch.mean(self.X_hat,[0])[0]
                                - torch.mean(self.X,[0])[0]))
        S_loss = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])

        self.err_g = G_adv + self.opt.w_g * (V1 + V2) + torch.sqrt(S_loss)
        self.err_g.backward(retain_graph=True)

        BaseModel.loss_history["g"].append(self.err_g.item())
        print("Loss G: ", self.err_g)


    def backward_s(self):
      """ Backpropagate through netS
      """
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)
      BaseModel.loss_history["s"].append(self.err_s.item())

      print("Loss S: ", self.err_s)
   #   print(torch.autograd.grad(self.err_s, self.nets.parameters()))

    def backward_d(self):
        # ==========================
        #   WGAN-Critic Loss (no BCE!)
        # ==========================

        # Real and fake latent states
        H_real = self.H                                  # from Encoder(Xc)
        H_fake = self.H_hat.detach()                     # from Supervisor(Generator)

        # Concatenate condition
        H_real_c = torch.cat([H_real, self.C], dim=-1)
        H_fake_c = torch.cat([H_fake, self.C], dim=-1)

        # Critic outputs
        D_real = self.netd(H_real_c).mean()
        D_fake = self.netd(H_fake_c).mean()

        # Gradient Penalty
        gp = self.grad_penalty(H_real, H_fake, self.C)

        lambda_gp = 10.0
        self.err_d = (D_fake - D_real) + lambda_gp * gp

        self.err_d.backward(retain_graph=True)
        BaseModel.loss_history["d"].append(self.err_d.item())


        

     # print("Loss D: ", self.err_d)

    def optimize_params_er(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_er()

      # Backward-pass
      # nete & netr
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_er_(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_er()
      self.forward_s()
      # Backward-pass
      # nete & netr
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er_()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_s(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_s()

      # Backward-pass
      # nets
      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()

    def optimize_params_g(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_s()
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_g.zero_grad()
      self.optimizer_s.zero_grad()
      self.backward_g()
      self.optimizer_g.step()
      self.optimizer_s.step()

    def optimize_params_d(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_d()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()
