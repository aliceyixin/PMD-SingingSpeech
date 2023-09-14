import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, hidden_size=10):
        super(MINE, self).__init__()
        self.layers= nn.Sequential(nn.Linear(256, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size,hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1)
                                  )

    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x,], dim=0)
        idx = torch.randperm(batch_size)
        
        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = -(torch.mean(pred_xy) 
               - torch.log(torch.mean(torch.exp(pred_x_y))))

        return loss