import os
import torch
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer import Trainer

train_stage = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dir = "dataset"
outputs_dir = "outputs/results/training"
weights_dir = "weights"
G_filename = "Generator_model_perceptual_real_color_70_512_9_.pth"
D_finename = "Discriminator_model_perceptual_real_color_70_512_9_.pth"
G_path = os.path.join(weights_dir,G_filename)
D_path = os.path.join(weights_dir,D_finename)
lambda_L1 = 100
lambda_Per = 1
lambda_Col = 50
lambda_Con = 1
lambda_Sty = 1
lambda_HSV = 1 
lambda_YUV = 1
num_epochs = 1
G = Generator()
D = Discriminator()
G.to(device)
D.to(device)
G = torch.load(G_path)
D = torch.load(D_path)

train=Trainer(device=device, dataset_dir=dataset_dir, outputs_dir=outputs_dir, 
              G=G, D=D, lambda_L1=lambda_L1, lambda_Per=lambda_Per, 
              lambda_Col=lambda_Col, lambda_Con=lambda_Con, lambda_Sty=lambda_Sty, 
              lambda_HSV=lambda_HSV, lambda_YUV=lambda_YUV)
train.fit(num_epochs=num_epochs)

torch.save(train.G,f"Generator_model_sketch_2_anime_stage_{train_stage}.pth")
torch.save(train.D,f"Discriminator_model_sketch_2_anime_stage_{train_stage}.pth")