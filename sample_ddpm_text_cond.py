
import torch
import torchvision
import argparse
import yaml
import os
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 
from torchvision import transforms 
from models.unet_cond_base import Unet
#from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.text_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_tensor_as_heatmap(tensor, title="Heatmap", xtick_interval=10, ytick_interval=10):
    if tensor.dim() != 3:
        raise ValueError("El tensor debe tener tres dimensiones (batch_size, height, width).")

    batch_size = tensor.size(0)
    height = tensor.size(1)
    width = tensor.size(2)

    # Determinar el número de columnas y filas para la cuadrícula
    num_cols = min(5, batch_size)
    num_rows = (batch_size + num_cols - 1) // num_cols
    
    title = f"/content/heatmap_{title}.png"

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = np.array(axes).reshape(-1)



    for i in range(batch_size):
        tensor_np = tensor[i].cpu().detach().numpy()
        im = axes[i].imshow(tensor_np, cmap="viridis", aspect='auto')

        axes[i].set_title(f"Slice {i + 1}")

        # Configurar los ticks en intervalos
        axes[i].set_xticks(np.arange(0, width, xtick_interval))
        axes[i].set_yticks(np.arange(0, height, ytick_interval))

        # Mostrar la barra de color a la derecha del heatmap
        fig.colorbar(im, ax=axes[i])

    # Desactivar ejes vacíos si el número de slices es menor que el número de subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title, fontsize=6)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(title)
    #plt.show()



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
    
    

    # transform = transforms.Compose([transforms.ToTensor()]) 

    # image_path  = "/content/paint.jpg"
    # image= Image.open(image_path) 
    # original = transform(image).unsqueeze(0) 
    # noise = torch.randn_like(original)  
    # t = torch.randint(0,1000, (1,))  
    

     
    # value1, value2 = scheduler.sample_prev_timestep(original, noise, t) 


    # value = value.float()
    # value_min_val = torch.min(value) 
    # value_max_val = torch.max(value)

    # value2 = value2.float()
    # value2_min_val = torch.min(value2) 
    # value2_max_val = torch.max(value2)

    # value3 = value3.float()
    # value3_min_val = torch.min(value3) 
    # value3_max_val = torch.max(value3)

    # normalized_value = (value - value_min_val) / (value_max_val - value_min_val) 
    # normalized_value2 = (value2 - value2_min_val) / (value2_max_val - value2_min_val) 
    # normalized_value3 = (value3 - value3_min_val) / (value3_max_val - value3_min_val) 


    # print(value.shape)
    # print(value2.shape)
    # print(value3.shape) 

    # visualize_tensor_as_heatmap(normalized_value.squeeze(0),"Heatmap")
    # visualize_tensor_as_heatmap(normalized_value2.squeeze(0),"Heatmap2")
    # visualize_tensor_as_heatmap(normalized_value3.squeeze(0),"Heatmap3")

    text_tokenizer = None
    text_model = None



    condition_config = get_config_value(diffusion_model_config, key="condition_config",default_value=None)
    assert condition_config is not None, ("this sampling script is for text conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, "condition_types", [])
    assert "text" in condition_types, ("this sampling script is for text conditional"
                                      "but no text condition found in config")
    validate_text_config(condition_config)

    with torch.no_grad():

         text_tokenizer, text_model = get_tokenizer_and_model(condition_config["text_condition_config"]
                                                              ["text_embed_model"], device=device)

    model = Unet(im_channels=autoencoder_model_config["z_channels"], 
                 model_config=diffusion_model_config).to(device) 

    model.eval() 
    print(os.path.join(train_config["task_name"],train_config["ldm_ckpt_name"]))
    
    if os.path.exists(os.path.join(train_config["task_name"], 
                      train_config["ldm_ckpt_name"])):    
         print("Loaded unet checkpoint")  
         model.load_state_dict(torch.load(os.path.join(train_config["task_name"], 
                                                       train_config["ldm_ckpt_name"]), 
                                          map_location=device))  
    else: 
        raise Exception("Model checkpoing {} not found", format(os.path.join(train_config["task_name"],
                                                                              train_config["ldm_ckpt_name"]))) 
        
    # if not os.path.exists(train_config["task_name"]):   
    #     os.mkdir(train_config["task_name"])   

    # vae = VQVAE(in_channels=dataset_config["in_channels"],
    #             model_config=autoencoder_model_config).to(device)
    # vae.eval()  

    # if os.path.exists(os.path.join(train_config['task_name'],
    #                                train_config['vqvae_autoencoder_ckpt_name'])):
    #     print('Loaded vae checkpoint')
    #     vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                 train_config['vqvae_autoencoder_ckpt_name']),
    #                                    map_location=device), strict=True)
    # else:
    #     raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
    #                                                                       train_config['vqvae_autoencoder_ckpt_name'])))

    # with torch.no_grad(): 
    #     sample(model, scheduler, train_config, diffusion_model_config, 
    #             autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model) 
         



parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with only '
                                                 'text conditioning')

parser.add_argument('--config', dest='config_path',
                     default='config/celebhq_text_cond.yaml', type=str)
args = parser.parse_args()


infer(args)

