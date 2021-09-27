import torch
import matplotlib.pylab as plt
import os
import soundfile
import glob

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")

def scan_checkpoint(cp_dir, prefix):
    if not os.path.isdir(cp_dir):
        return None
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
    
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()
    return fig

def save_audio(tensor, path_prefix, indices, steps, sampling_rate=24000):
    filepath = f"{path_prefix}_{steps:08}_{indices:02}.wav"
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    soundfile.write(filepath, tensor.cpu().numpy().flatten(), sampling_rate)


