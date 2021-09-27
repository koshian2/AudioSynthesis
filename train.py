import json
from re import I
from attrdict import AttrDict
from numpy import lib
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import itertools
import os
import time
from losses import discriminator_loss, feature_loss, generator_loss, mel_spectrogram_loss
from models import MultiPeriodDiscriminator, MultiScaleDiscriminator, SpectrogramDiscriminator, Generator
from utils import plot_spectrogram, save_audio, save_checkpoint, scan_checkpoint, load_checkpoint
from data.data_loader import PretrainedDataset

def train():
    with open("config.json", encoding="utf-8") as fp:
        h = json.load(fp)
    with open("arguments.json", encoding="utf-8") as fp:
        args = json.load(fp)

    # stage別の選択
    h["batch_size"] = h["stage_batch_sizes"][args["stage"]-1]
    args["training_epochs"] = args["stage_training_epochs"][args["stage"]-1]
    h, args = AttrDict(h), AttrDict(args)

    # CPU/GPUデバイスの取得
    device = torch.device(f"cuda:{h.gpu_ids[0]}") if len(h.gpu_ids) > 0 else torch.device("cpu")

    # モデルの初期化
    generator = Generator(h, args).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    spd = SpectrogramDiscriminator().to(device)

    # 係数の復元
    if args.restore_checkpoint:
        cp_g = scan_checkpoint(args.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(args.checkpoint_path, 'do_')
    if args.restore_checkpoint and cp_g is not None and cp_do is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        spd.load_state_dict(state_dict_do['spd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        print("Model restored.")
        print(f"steps:{steps}, last_epoch:{last_epoch}")
    else:
        steps = 0
        last_epoch = -1
        state_dict_do = None

    if len(h.gpu_ids) > 1:
        generator = DataParallel(generator, h.gpu_ids).to(device)
        mpd = DataParallel(mpd, h.gpu_ids).to(device)
        msd = DataParallel(msd, h.gpu_ids).to(device)
        spd = DataParallel(spd, h.gpu_ids).to(device)

    optim_g = torch.optim.Adam(generator.parameters(), h.learning_rate_g, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(itertools.chain(msd.parameters(), mpd.parameters(), spd.parameters()),
                                h.learning_rate_d, betas=[h.adam_b1, h.adam_b2])
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    print(f"+ stage:{args.stage}, batch_size:{h.batch_size}, epochs_to_train:{args.training_epochs} +")

    sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))
    train_loader = PretrainedDataset(h, args, "train")
    validation_loader = PretrainedDataset(h, args, "test")
    train_loader = DataLoader(train_loader, batch_size=h.batch_size, num_workers=h.num_workers, drop_last=True,
                              shuffle=True, collate_fn=train_loader.collate_fn)
    validation_loader = DataLoader(validation_loader, batch_size=h.batch_size, num_workers=0,
                              shuffle=False, collate_fn=validation_loader.collate_fn)
    
    generator.train()
    mpd.train()
    msd.train()
    spd.train()
    optim_d.zero_grad()
    optim_g.zero_grad()

    # ToDo epoch回りをもう少し丁寧に定義する
    for epoch in range(max(0, last_epoch), args.training_epochs):
        start = time.time()
        for i, batch in enumerate(train_loader):
            x, y, y_mel, mel_mask = batch
            #print("batch_shape", [xx.shape for xx in batch])

            x = [torch.autograd.Variable(xx.to(device, non_blocking=True)) if xx is not None else None for xx in x ]
            if args.stage >= 2:
                y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))

            y_g_hat_mel, y_g_hat = generator(x[0], x[1:])
            update_g = (steps // h.batch_size) % h.update_d_steps_per_g == 0

            # train Discriminators
            if args.stage >= 2:
                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            # Spectrogram
            y_dm_hat_r, y_dm_hat_g, _, _ = spd(torch.unsqueeze(y_mel, 1), torch.unsqueeze(y_g_hat_mel.detach(), 1))
            loss_disc_m, losses_disc_m_r, losses_disc_m_g = discriminator_loss(y_dm_hat_r, y_dm_hat_g)

            if args.stage == 1:
                loss_disc_all = loss_disc_m
            else:
                loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_m
            (loss_disc_all * h.batch_size / h.accumulation_steps).backward() # accumeration

            # train Generator
            # L1 Mel-Spectrogram Loss
            loss_mel = mel_spectrogram_loss(y_mel, y_g_hat_mel, mel_mask) * 40
            loss_mel += mel_spectrogram_loss(y_mel, y_g_hat_mel, 1-mel_mask) * 5

            if args.stage >= 2:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            y_dm_hat_r, y_dm_hat_g, fmap_m_r, fmap_m_g = spd(torch.unsqueeze(y_mel, 1), torch.unsqueeze(y_g_hat_mel, 1))

            if args.stage >= 2:
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_fm_m = feature_loss(fmap_m_r, fmap_m_g)

            if args.stage >= 2:
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_m, losses_gen_m = generator_loss(y_dm_hat_g)

            if args.stage >= 2:
                loss_gen_all = loss_gen_s + loss_gen_f + loss_gen_m + loss_fm_s + loss_fm_f + loss_fm_m + loss_mel
            else:
                loss_gen_all = loss_gen_m + loss_fm_m + loss_mel

            (loss_gen_all * h.batch_size / h.accumulation_steps).backward() # accumeration

            # Gradient Accumeration
            if steps % h.accumulation_steps < h.batch_size and steps != 0:
                optim_d.step()
                if update_g:
                    optim_g.step()
                optim_d.zero_grad()
                optim_g.zero_grad()

            # checkpointing
            if steps % args.checkpoint_interval < h.batch_size: #and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(args.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': (generator.module if len(h.gpu_ids) > 1 else generator).state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(args.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, 
                                {'mpd': (mpd.module if len(h.gpu_ids) > 1
                                                        else mpd).state_dict(),
                                    'msd': (msd.module if len(h.gpu_ids) > 1
                                                        else msd).state_dict(),
                                    'spd': (spd.module if len(h.gpu_ids) > 1
                                                        else spd).state_dict(),
                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                    'epoch': epoch})

            # Tensorboard summary logging
            if steps % args.summary_interval < h.batch_size:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                        format(steps, loss_gen_all, mel_error, time.time() - start))
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)

            # Validation
            if steps % args.validation_interval < h.batch_size:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0

                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, y_mel, mel_mask = batch
                        x = [torch.autograd.Variable(xx.to(device, non_blocking=True)) if xx is not None else None for xx in x ]
                        y_g_hat_mel, y_g_hat = generator(x[0], x[1:])
                        val_err_tot += F.l1_loss(y_mel.to(device), y_g_hat_mel).item()

                        if j <= 4:
                            if steps == 0:
                                # SummaryWriterのadd_audioが異常に重いので、ファイルとして保存する
                                if args.stage >= 2:
                                    save_audio(y[0], f'{args.checkpoint_path}/gt/y', j, steps, h.sampling_rate)
                                sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].numpy()), steps)

                            if args.stage >= 2:
                                save_audio(y_g_hat[0], f'{args.checkpoint_path}/generated/y_hat', j, steps, h.sampling_rate)
                            sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                            plot_spectrogram(y_g_hat_mel[0].cpu().numpy()), steps)

                    val_err = val_err_tot / (j+1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)

                generator.train()

            steps += h.batch_size

        # end of epoch
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
        #if (epoch-last_epoch) >= 2:
        #    break

if __name__ == "__main__":
    train()
