import torch


def getTimeEmbedding(timeSteps, time_embedding):
    '''
    This Time EMbedding Block takes 1d of batch size time and outputs the time embedding
    of (batch_size, time_embedding)

    sin(pos/10000^(2i/d_model))
    cos(pos/10000^(2i/d_model))

    '''
    factor = 10000 ** (torch.arange(start=0, end=time_embedding//2, device=timeSteps.device)/ (time_embedding//2))
    time_embed = timeSteps[:, None].repeat(1, time_embedding//2)/factor
    time_embed = torch.cat([torch.sin(time_embed), torch.cos(time_embed)], dim=-1)

    return time_embed

    