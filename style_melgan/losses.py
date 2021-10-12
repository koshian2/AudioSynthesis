import torch
import torch.nn.functional as F

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    n = len(disc_real_outputs)

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(F.relu(1-dr[-1])) # last hidden layer
        g_loss = torch.mean(F.relu(1+dg[-1]))
        loss += (r_loss + g_loss) / n
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    n = len(disc_outputs)

    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(-dg[-1])
        gen_losses.append(l.item())
        loss += l / n

    return loss, gen_losses

def pqmf_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    n = len(disc_real_outputs)

    pqmr_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        l = F.l1_loss(dr[0], dg[0])
        pqmr_losses.append(l.item())
        loss += l / n
    return loss, pqmr_losses


# うまくいかないのでHiFi-GANのロスを使う

# def discriminator_loss(disc_real_outputs, disc_generated_outputs):
#     loss = 0
#     r_losses = []
#     g_losses = []
#     for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
#         r_loss = torch.mean((1-dr[-1])**2)
#         g_loss = torch.mean(dg[-1]**2)
#         loss += (r_loss + g_loss)
#         r_losses.append(r_loss.item())
#         g_losses.append(g_loss.item())

#     return loss, r_losses, g_losses

# def generator_loss(disc_outputs):
#     loss = 0
#     gen_losses = []
#     for dg in disc_outputs:
#         l = torch.mean((1-dg[-1])**2)
#         gen_losses.append(l)
#         loss += l

#     return loss, gen_losses

def feature_matching_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    fmap_losses = []
    for reals, gens in zip(disc_real_outputs, disc_generated_outputs):
        for dr, dg in zip(reals, gens):
            loss += F.l1_loss(dr, dg)            
            fmap_losses.append(loss.item())
    return loss, fmap_losses
