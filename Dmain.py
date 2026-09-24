from Cdomain import (subdomain3, subdomain2, subdomain1)
from Aconfig import (DEVICE, Config)
from Bmodel import (PINN1, PINN2, PINN3)
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn




config = Config()


x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.x_max

x_min.to(DEVICE), x_max.to(DEVICE), t_min.to(DEVICE), t_max.to(DEVICE)

t_mid = (t_max + t_min)/2

t_mid.to(DEVICE)

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)

model3 = PINN3().to(DEVICE)


N = 7500

# -------------------------------------------------------------------------------------------------
# SUBDOMAIN POINTS GENERATIONS
# -------------------------------------------------------------------------------------------------


# Subdomain1 points generation
x1_min, x1_max, t1_min, t1_max = subdomain1()

x1 = x1_min + (x1_max - x1_min) * torch.rand(N,1)
t1 = t1_min + (t1_max - t1_min) * torch.rand(N,1)

# subdomain2 points generation
x2_min, x2_max, t2_min, t2_max = subdomain2()

x2 = x2_min + (x2_max - x2_min) * torch.rand(N,1)
t2 = t2_min + (t2_max - t2_min) * torch.rand(N,1)



# subdomain5 points generation
x3, t3 = subdomain3()

# -------------------------------------------------------------------------------------------------
# SUBDOMAIN LOSS FUNCTION FOR EACH SUBDOMAINS
# -------------------------------------------------------------------------------------------------


def Subdomain1Loss():

    x = x1
    t = t1
    x_i = torch.linspace(x1_min,x1_max, steps=2000, device= DEVICE).reshape(-1,1)
    t_i = torch.zeros_like(x_i)

    t_b_left = torch.linspace(t1_min,t1_max, steps=400, device= DEVICE).reshape(-1,1)
    x_b_left = torch.zeros_like(t_b_left)
    

    
    t_b_right = torch.linspace(t1_min,t1_max, steps=400, device= DEVICE).reshape(-1,1)
    x_b_right = torch.ones_like(t_b_right)

    x_b_left.requires_grad_(True)
    t_b_left.requires_grad_(True)

    x_b_right.requires_grad_(True)
    t_b_right.requires_grad_(True)

    x_i.requires_grad_(True)
    t_i.requires_grad_(True)

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred1 = model1(x,t)
    u_pred_i = model1(x_i, t_i)
    u_pred_left_b = model1(x_b_left,t_b_left)
    u_pred_right_b = model1(x_b_right, t_b_right)

    InitialLoss = nn.MSELoss()(u_pred_i, torch.sin(torch.pi * x_i))

    BoundaryLossLeft = nn.MSELoss()(u_pred_left_b, torch.zeros_like(u_pred_left_b))

    BoundaryLossRight = nn.MSELoss()(u_pred_right_b, torch.zeros_like(u_pred_left_b))

    u_t = torch.autograd.grad(u_pred1, t, grad_outputs= torch.ones_like(u_pred1), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred1, x, grad_outputs= torch.ones_like(u_pred1), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss1 = Pdeloss + InitialLoss + BoundaryLossLeft + BoundaryLossRight

    return loss1


def Subdomain2Loss():

    x = x2
    t = t2

    
    t_b_left = torch.linspace(t2_min, t2_max , steps=400, device= DEVICE).reshape(-1,1)
    x_b_left = torch.zeros_like(t_b_left)

    t_b_right = torch.linspace(t2_min, t2_max, steps=400, device=DEVICE).reshape(-1,1)
    x_b_right = torch.ones_like(t_b_right)
    

    x_b_left.requires_grad_(True)
    t_b_left.requires_grad_(True)

    x_b_right.requires_grad_(True)
    t_b_right.requires_grad_(True)

   

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred2 = model2(x,t)

    u_pred_left_b = model2(x_b_left, t_b_left)

    u_pred_right_b = model2(x_b_right,t_b_right)

    

    BoundaryLossLeft = nn.MSELoss()(u_pred_right_b, torch.zeros_like(u_pred_right_b))

    BoundaryLossRight = nn.MSELoss()(u_pred_left_b, torch.zeros_like(u_pred_right_b))

    u_t = torch.autograd.grad(u_pred2, t, grad_outputs= torch.ones_like(u_pred2), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred2, x, grad_outputs= torch.ones_like(u_pred2), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss2 = Pdeloss + BoundaryLossLeft + BoundaryLossRight

    return loss2



def Subdomain3Loss():

    x = x3
    t = t3

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred3 = model3(x,t)

    u_t = torch.autograd.grad(u_pred3, t, grad_outputs= torch.ones_like(u_pred3), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred3, x, grad_outputs= torch.ones_like(u_pred3), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss5 = Pdeloss 

    return loss5

optimizer1 = torch.optim.Adam(model1.parameters(), lr= config.learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr= config.learning_rate)
optimizer3 = torch.optim.Adam(model3.parameters(), lr= config.learning_rate)




for epoch in range(config.num_epochs):
    optimizer1.zero_grad()
    loss1 = Subdomain1Loss()
    loss1.backward()
    optimizer1.step()


    optimizer2.zero_grad()
    loss2 = Subdomain2Loss()
    loss2.backward()
    optimizer2.step()
    
    optimizer3.zero_grad()
    loss3 = Subdomain3Loss()
    loss3.backward()
    optimizer3.step()

    

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss1: {loss1.item():.6e} | Loss2: {loss2.item():.6e} | Loss3: {loss3.item():.6e} ")


print("Training finished")


torch.save(model1.state_dict(), "model1.pth")
torch.save(model2.state_dict(), "model2.pth")
torch.save(model3.state_dict(), "model3.pth")

