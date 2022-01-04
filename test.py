import os
import torch
from src.models.generator import Generator
from src.testing.tester import Tester

test_dir = "test"
outputs_dir = "outputs/results/testing"
weights_dir = "weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_filename = "Generator_model_perceptual_real_color_70_512_9_.pth"
G_path = os.path.join(weights_dir,G_filename)
G = Generator()
G.to(device)
G = torch.load(G_path)

test = Tester(G=G, test_dir=test_dir, outputs_dir=outputs_dir)
test.fit()