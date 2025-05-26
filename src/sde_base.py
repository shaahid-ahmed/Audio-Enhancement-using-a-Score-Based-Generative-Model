from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

class SDEBase(nn.Module):
    def __init__(self, eps=1e-5, rescale=False):
        super().__init__()
        self.eps = eps
        self.rescale = rescale

    @property
    def T(self):
        return 1

    def drift_coef(self, x, y, t):
        pass
    
    def diffusion_coef(self, t):
        pass
    
    def x0_coef(self, t):
        pass
    
    def sigma_t(self, t):
        pass

    def _mean(self, x0, y, t):
        pass

    def _std(self, t):
        pass

    def marginal_prob(self, x0, y, t):
        pass

    def match_dim(self, x, y):
        while len(x.shape) < len(y.shape):
            x = x.unsqueeze(-1)
        return x

    def forward(self, model, x, y, t):
        F = torch.cat([x, y], dim=1) 
        # F = torch.cat((F[:,[0],:,:].real, F[:,[0],:,:].imag,
        #         F[:,[1],:,:].real, F[:,[1],:,:].imag), dim=1)
        score = model(F, t)
        return score*(-1)
    
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        s_t = self.sigma_t(t)
        x0_coef = self.x0_coef(t)
        s_t = self.match_dim(s_t, x_0)
        x0_coef = self.match_dim(x0_coef, x_0)
        
        return x0_coef * x_0 + s_t * noise
    
    def sample_time_batch(self, batch_size, device=None):
        return (self.eps - 1) * torch.rand(batch_size, device=device) + 1
    
    def discretize(self, x, y, t, stepsize):
        dt = torch.tensor(stepsize)
        drift = self.drift_coef(x, y, t)
        diffusion = self.diffusion_coef(t)
        f = drift * dt
        G = diffusion * torch.sqrt(dt)
        return f, G
    
    
    def reverse_discretize(self, model, x, y, t, step_size):
        f, G = self.discretize(x, y, t, step_size)
        rev_f = f - G[:, None, None, None] ** 2 * self.forward(model, x, y, t) * (0.5)
        rev_G = torch.zeros_like(G)
        return rev_f, rev_G

    
    def euler_maruyama_update(self, x, y, t, step_size, N=1000):
        dt = -1.0 / N
        z = torch.randn_like(x)
        f, g = self.discretize(x, y, t, step_size)
        x_mean = x + f * dt
        x = x_mean + g[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean
    
    
    
    def ReverseDiffusionPredictor(self, model, x, y, t, step_size):
        f, g = self.reverse_discretize(model, x, y, t, step_size)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean

    
    def langevin_step(self, model, x, y, t, snr=0.1):
        score = self.forward(model, x, y, t)
        noise = torch.randn_like(x)
        score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = ((snr * noise_norm / score_norm) ** 2 * 2).unsqueeze(0)

        x_mean = x + step_size[:, None, None, None] * score
        x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]
        return x, x_mean

    
    def predictor_corrector_step(self, model, x, y, t, delta_t, n_lang_steps=1, snr=0.1):
        for _ in range(n_lang_steps):
            x = self.langevin_step(model, x, y, t, snr)

        x, _ = self.euler_maruyama_update(x, y, t, delta_t)
        return x

    
    def predictor_corrector_sample(self, model, shape, device, y, n_steps=500, n_lang_steps=1, snr=0.1):
        with torch.no_grad():
            std = self._std(torch.ones((y.shape[0],),device=y.device))
            xt = y + torch.randn_like(y) * std[:, None, None, None]
            timesteps = torch.linspace(1, self.eps, n_steps, device=device)  # Ensure self.eps is defined elsewhere
            # delta_t = time_steps[0] - time_steps[1]
            for i in tqdm(range(n_steps)):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1] # from eps to 0
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = self.langevin_step(model,xt, y, vec_t)
                xt, xt_mean = self.ReverseDiffusionPredictor(model,xt, y, vec_t, stepsize)
                # time_batch = torch.ones(shape[0], device=device) * t
                # x_t = self.predictor_corrector_step(model, x_t, y, time_batch, delta_t, n_lang_steps, snr)
            x_result = xt_mean
            # ns = n_steps * (n_lang_steps + 1)
            return xt