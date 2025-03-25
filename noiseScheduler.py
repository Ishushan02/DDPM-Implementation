import torch

class LinearNoiseScheduler:
    '''
    In this function according to given formula of alpha, beta, 
    the cummulative product of alpha, we store those values and 
    also add noise to the given image with the above factors.
    We are storing it because we can directly use these values while in sampling procedure
    See the formula dewscription for detailed summary.
    '''
    def __init__(self, timeSteps, betaStart, betaEnd):
        self.timeSteps = timeSteps
        self.betaStart = betaStart
        self.betaEnd = betaEnd

        self.betas = torch.linspace(betaStart, betaEnd, timeSteps)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

    def add_noise(self, image, noise, t):
        '''
        This is the forward function of adding noise to the image
        at timestep t, with certain noise.
        The image dimension would be (batch_size, channel, height, width)

        See the formula that we are applying for the forward process of adding Noise at time t.
        '''
        batch_size, channel, height, width = image.shape

        # this becomes batch, 1, 1, 1
        sqrt_alphar_hat = self.sqrt_alpha_hat[t].reshape(batch_size)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].reshape(batch_size)

        for _ in range(len(image.shape) -1):
            sqrt_alphar_hat = sqrt_alphar_hat.unsqueeze(-1)
            sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.unsqueeze(-1)

        return sqrt_alphar_hat * image + sqrt_one_minus_alpha_hat * noise
    
    
    def sampling_step(self, image_t, noise_pred, t):
        '''
        We get image at time t, and also the predicted noise at time t
        Here in this step basically we get predicted noise from the Neural Network
        we have to remove it, we remove it.
        See the formula that we are applying for the sampling process of at time t.
        '''
        image = (image_t - (self.sqrt_one_minus_alpha_hat[t] * noise_pred))/ self.sqrt_alpha_hat
        image = torch.clamp(image, min=-1., max=1.)

        mean = (1/torch.sqrt(self.alphas[t])) * (image_t - ((self.betas[t] * noise_pred)/self.sqrt_one_minus_alpha_hat[t]))

        if t==0:
            # when t is 0, no variance just return mean and the image
            return mean, image
        
        # else valvulate variance using the formula and send it and send the gaussiaan of it
        variance = (self.betas[t]) * (1 - self.alpha_hat[t-1])/(1 - self.alpha_hat[t])
        sigma = variance ** 0.5
        z = torch.randn(image_t.shape).to(image_t.device)

        return  mean + sigma * z, image





