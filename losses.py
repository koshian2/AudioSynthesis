import torch
import torch.nn.functional as F

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for rl, gl in zip(fmap_r, fmap_g):
        loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def mel_spectrogram_loss(y_mel, y_g_hat_mel, mel_mask):
    return F.l1_loss(y_mel, y_g_hat_mel, reduction="sum") / (torch.sum(mel_mask) + 1e-8)

if __name__ == "__main__":
    from test_code.model_test import check_train_loop
    check_train_loop()    