
import torch
import torchvision 
import argparse 
import yaml 
import os 
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.text_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def infer(args): 
    with open(args.config_path, "r") as file: 
        try: 
            config = yaml.safe_load(file) 
        except yaml.YAMError as exc: 
            print(exc)  

    diffusion_config = config["diffusion_params"] 
    dataset_config = config["dataset_params"]
    diffusion_model_config = config["ldm_params"] 
    autoencoder_model_config = config["autoencoder_params"] 
    train_config = config["train_params"]

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config["num_timesteps"], 
                                     beta_start = diffusion_config["beta_start"], 
                                     beta_end = diffusion_config["beta_end"])
    
  
    text_tokenizer = None
    text_model = None 



    condition_config = get_config_value(diffusion_model_config, key="condition_config",default_value=None)
    assert condition_config is not None, ("this sampling script is for text conditional " 
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, "condition_types", []) 
    assert "text" in condition_types, ("this sampling script is for text conditional" 
                                      "but no text condition found in config") 
    validate_text_config(condition_config)




parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with only '
                                                 'text conditioning')

parser.add_argument('--config', dest='config_path',
                     default='config/celebhq_text_cond.yaml', type=str)
args = parser.parse_args()


infer(args)

