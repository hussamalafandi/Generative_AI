import datetime


def generate_run_name(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.get("run_name", f"DCGAN_CelebA_lr{config['lr']}_bs{config['batch_size']}_{timestamp}")
