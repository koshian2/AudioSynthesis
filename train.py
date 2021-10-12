import json
from attrdict import AttrDict
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from style_melgan.style_melgan_models import StyleMelGANGenerator, StyleMelGANDiscriminator, SpectrogramDiscriminator
from style_melgan.style_melgan_layers import MoraEncoder, LogMelSpectrogram
from style_melgan.losses import generator_loss, discriminator_loss, pqmf_loss
from utils import plot_spectrogram, save_audio, save_checkpoint, scan_checkpoint, load_checkpoint
from data.data_loader import PretrainedDataset

def train():
    with open("style_melgan/config.json", encoding="utf-8") as fp:
        h = AttrDict(json.load(fp))

    # CPU/GPUデバイスの取得
    device_g = torch.device(f"cuda:{h.gpu_id_g}") if h.gpu_id_g >= 0 else torch.device("cpu")
    device_d = torch.device(f"cuda:{h.gpu_id_d}") if h.gpu_id_d >= 0 else torch.device("cpu")

    # モデルの初期化
    generator = StyleMelGANGenerator(h).to(device_g, non_blocking=True)
    discriminator = StyleMelGANDiscriminator(h).to(device_d, non_blocking=True)
    spectrogram_d = SpectrogramDiscriminator().to(device_g, non_blocking=True)
    mora_encoder = MoraEncoder(h, device_g, device_d).to(device_g, non_blocking=True) # ここtoいる？
    mel_extractor = LogMelSpectrogram(h).to(device_d, non_blocking=True)

    # 係数の復元
    if h.restore_checkpoint:
        cp_g = scan_checkpoint(h.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')
    if h.restore_checkpoint and cp_g is not None and cp_do is not None:
        state_dict_g = load_checkpoint(cp_g, device_g)
        state_dict_do = load_checkpoint(cp_do, device_d)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        spectrogram_d.load_state_dict(state_dict_do["spectrogram_d"])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        print("Model restored.")
        print(f"steps:{steps}, last_epoch:{last_epoch}")
    else:
        steps = 0
        last_epoch = -1
        state_dict_do = None

    optim_g = torch.optim.Adam(generator.parameters(), h.learning_rate_g_initial, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(itertools.chain(discriminator.parameters(), spectrogram_d.parameters()), 
                                h.learning_rate_d, betas=[h.adam_b1, h.adam_b2])
    
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))
    train_loader = PretrainedDataset(h, "train")
    validation_loader = PretrainedDataset(h, "test")
    train_loader = DataLoader(train_loader, batch_size=h.batch_size, num_workers=h.num_workers, drop_last=True,
                              shuffle=True, collate_fn=train_loader.collate_fn)
    validation_loader = DataLoader(validation_loader, batch_size=h.batch_size, num_workers=0,
                              shuffle=False, collate_fn=validation_loader.collate_fn)
    
    generator.train()
    discriminator.train()
    optim_d.zero_grad()
    optim_g.zero_grad()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        start = time.time()
        for i, batch in enumerate(train_loader):
            if steps == h.adversarial_training_start_step:
                for param_group in optim_g.param_groups:
                        param_group['lr'] = h.learning_rate_g
                print(f"Decreased learning rate D from {h.learning_rate_g_initial} to {h.learning_rate_g}")
            
            x, y = mora_encoder(batch[0], pad_pre=h.sampling_rate//2, pad_post=h.sampling_rate//2)

            #print("batch_shape",x.shape, y.shape)
            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)

            enable_adversarial_training = steps >= h.adversarial_training_start_step

            y_g_hat = generator(x).to(device_d, non_blocking=True)
            y_mel = mel_extractor(y)
            y_g_hat_mel = mel_extractor(y_g_hat)

            ## Train discriminator
            if enable_adversarial_training:
                y_df_hat_r, y_df_hat_g = discriminator(y, y_g_hat.detach())
                loss_disc_wave, losses_disc_f_r, losses_disc_f_g  = discriminator_loss(y_df_hat_r, y_df_hat_g)

                y_ds_hat_r, y_ds_hat_g, fmap_rs, fmap_gs= spectrogram_d(y_mel.unsqueeze(1), y_g_hat_mel.detach().unsqueeze(1))
                loss_disc_spect, losses_disc_s_r, losses_disc_s_g  = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_wave + loss_disc_spect
                (loss_disc_all * h.batch_size / h.accumulation_steps).backward()

            ## Train Generator
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * h.lambda_loss_mel
            # 波形の中心値が0に近づくように正則化
            loss_mean_reg = torch.abs(torch.mean(y_g_hat)) * h.lambda_loss_mean_reg

            if enable_adversarial_training:
                y_df_hat_r, y_df_hat_g = discriminator(y, y_g_hat)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

                y_ds_hat_r, y_ds_hat_g, fmap_rs, fmap_gs= spectrogram_d(y_mel.unsqueeze(1), y_g_hat_mel.unsqueeze(1))
                loss_gen_s, losses_gen_s_g  = generator_loss(y_ds_hat_g)

                # PQMRの値でもfeature matching
                loss_pqmr_f, losses_pqmr_f = pqmf_loss(y_df_hat_r, y_df_hat_g)
                loss_pqmr_f *= h.lambda_loss_pqmr_f

            else:
                loss_gen_f, loss_gen_s, loss_pqmr_f = 0, 0, 0

            #print(f"mel={loss_mel}, pqmr={loss_pqmr_f}, gen_f={loss_gen_f}, gen_s={loss_gen_s}, mean_reg={loss_mean_reg}")
            # 初期で120 / 5 / -0.5 / 0.5　ぐらい

            loss_gen_all = loss_mel  + loss_gen_f + loss_gen_s + loss_pqmr_f + loss_mean_reg
            (loss_gen_all * h.batch_size / h.accumulation_steps).backward()

            # Gradient Accumeration
            if (steps + 1) % h.accumulation_steps < h.batch_size:
                optim_d.step()
                optim_g.step()
                optim_d.zero_grad()
                optim_g.zero_grad()

            # checkpointing
            if steps % h.checkpoint_interval < h.batch_size: #and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': (generator).state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, 
                                {'discriminator': (discriminator).state_dict(),
                                 'spectrogram_d': (spectrogram_d).state_dict(),
                                 'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                 'epoch': epoch})

            # Tensorboard summary logging
            if steps % h.summary_interval < h.batch_size:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                        format(steps, loss_gen_all, mel_error, time.time() - start))
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)
                sw.add_scalar("training/gen_loss_f", loss_gen_f, steps)
                sw.add_scalar("training/gen_loss_s", loss_gen_s, steps)

            # Validation
            if steps % h.validation_interval < h.batch_size:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0

                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y = mora_encoder(batch[0], pad_pre=None, pad_post=None)
                        y_g_hat = generator(x).to(device_d, non_blocking=True)

                        y_mel = mel_extractor(y)
                        y_g_hat_mel = mel_extractor(y_g_hat)
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                        if j <= 4:
                            if steps == 0:
                                # SummaryWriterのadd_audioが異常に重いので、ファイルとして保存する
                                save_audio(y[0], f'{h.checkpoint_path}/gt/y', j, steps, h.sampling_rate)
                                sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu().numpy()), steps)

                            save_audio(y_g_hat[0], f'{h.checkpoint_path}/generated/y_hat', j, steps, h.sampling_rate)
                            sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                            plot_spectrogram(y_g_hat_mel[0].cpu().numpy()), steps)

                    val_err = val_err_tot / (j+1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)

                generator.train()

            steps += h.batch_size

        # end of epoch
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

if __name__ == "__main__":
    train()
