import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Définir votre architecture de réseau de neurones
class MonReseau(nn.Module):
	def __init__(self):
		super(MonReseau, self).__init__()
		self.mlp_extractor_policy_net_0  = nn.Linear(in_features=8, out_features=128)
		self.activation1 = nn.Tanh()
		self.action_net = nn.Linear(in_features=128, out_features=6)

	def forward(self, x):
		x = self.activation1(self.mlp_extractor_policy_net_0(x))
		x = self.action_net(x)
		return x

model = MonReseau()
# Charger le state_dict
state_dict = torch.load("policy.pth")

print(state_dict)

# Nouveau dictionnaire avec les clés filtrées
filtered_dict = {key: value for key, value in state_dict.items() if key.startswith("mlp_extractor.policy_net.0") or 
                                                                     key.startswith("mlp_extractor.policy_net.2") or
                                                                     key.startswith("action_net")}

nouveau_dict = {}
for cle, valeur in filtered_dict.items():
	if cle.startswith("mlp_extractor.policy_net.0.weight"):
		nouveau_dict["mlp_extractor_policy_net_0.weight"] = valeur
	elif cle.startswith("mlp_extractor.policy_net.0.bias"):
		nouveau_dict["mlp_extractor_policy_net_0.bias"] = valeur

	elif cle.startswith("mlp_extractor.policy_net.2.weight"):
		nouveau_dict["mlp_extractor_policy_net_2"] = valeur
	elif cle.startswith("mlp_extractor.policy_net.2.bias"):
		nouveau_dict["mlp_extractor_policy_net_2.bias"] = valeur
	else:
		nouveau_dict[cle] = valeur

# Affichage du nouveau dictionnaire
print("Nouveau dictionnaire avec les clés filtrées :")
for key, value in nouveau_dict.items():
    print(key, ":", value)

# Charger le nouveau state_dict dans le modèle
model.load_state_dict(nouveau_dict)

for name, param in model.named_parameters():
	if 'weight' in name:
		print(f'Layer: {name}, Shape: {param.shape}, Weights: {param}')
	elif 'bias' in name:
		print(f'Layer: {name}, Shape: {param.shape}, Biases: {param}')


first_layer_weights = model.mlp_extractor_policy_net_0.weight.t()
first_layer_biases = model.mlp_extractor_policy_net_0.bias
third_layer_weights = model.action_net.weight.t()
third_layer_biases = model.action_net.bias


f = open("Test/bestNNPPOcpp128.py.txt", "w")
f.write("{{")
for i in range(8):
	f.write("{")
	for j in range(128):
		f.write(str(first_layer_weights[i][j].detach().item()))
		if j < 127:f.write(",")
	f.write("}")
	if i < 7:f.write(",")
f.write("},{")
for i in range(128):
	f.write("{")
	for j in range(6):
		f.write(str(third_layer_weights[i][j].detach().item()))
		if j < 5:f.write(",")
	f.write("}")
	if i < 127:f.write(",")
f.write("}},\n{")
f.write("{")
for i in range(128):
	f.write(str(first_layer_biases[i].detach().item()))
	if i < 127:f.write(",")
f.write("},{")
for i in range(6):
	f.write(str(third_layer_biases[i].detach().item()))
	if i < 5:f.write(",")
f.write("}}\n")

f.close()

