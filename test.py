import os
import torch
from src.models.generator import Generator
from src.testing.tester import Tester


test_dir = "test"
outputs_dir = "outputs/results/testing"
weights_dir = "weights"
G_filename = "Generator_weights_anime_sketch_2_color.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_path = os.path.join(weights_dir,G_filename)
G = Generator()
G.to(device)
G.load_state_dict(torch.load(G_path))


test = Tester(G=G, test_dir=test_dir, outputs_dir=outputs_dir, device=device)
test.fit()