import tkinter as tk
import time 
import random
import math
import datetime
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchrl
from collections import deque
from helper import plot

from torch.distributions.categorical import Categorical

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
	Compose,
	DoubleToFloat,
	ObservationNorm,
	StepCounter,
	TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gym
from gym import spaces
from gym.utils import seeding

is_fork = multiprocessing.get_start_method() == "fork"
device = (
	torch.device(0)
	if torch.cuda.is_available() and not is_fork
	else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
	0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# Définir votre architecture de réseau de neurones
class MonReseau(nn.Module):
	def __init__(self):
		super(MonReseau, self).__init__()
		self.couche_lineaire1 = nn.Linear(8, 64)
		#self.dropout1 = nn.Dropout(p=0.1)  # Ajout de dropout à la première couche
		self.couche_lineaire2 = nn.Linear(64, 6)
		#self.dropout2 = nn.Dropout(p=0.1)  # Ajout de dropout à la deuxième couche
		#self.couche_lineaire3 = nn.Linear(100, 6)

	def forward(self, x):
		x = F.relu(self.couche_lineaire1(x))
		#x = self.dropout1(x)  # Dropout après la première couche linéaire
		x = self.couche_lineaire2(x)
		#x = self.dropout2(x)  # Dropout après la deuxième couche linéaire
		x = F.softmax(x, dim=-1)  # Calculer les probabilités avec softmax
		x = Categorical(x)  # Créer une distribution catégorique
		return x

class MonReseauV(nn.Module):
	def __init__(self):
		super(MonReseauV, self).__init__()
		self.couche_lineaire1 = nn.Linear(8, 64)
		#self.dropout1 = nn.Dropout(p=0.1)  # Ajout de dropout à la première couche
		self.couche_lineaire2 = nn.Linear(64, 1)
		#self.dropout2 = nn.Dropout(p=0.1)  # Ajout de dropout à la deuxième couche
		#self.couche_lineaire3 = nn.Linear(100, 6)

	def forward(self, x):
		x = F.relu(self.couche_lineaire1(x))
		#x = self.dropout1(x)  # Dropout après la première couche linéaire
		#x = F.relu(self.couche_lineaire2(x))
		#x = self.dropout2(x)  # Dropout après la deuxième couche linéaire
		x = self.couche_lineaire2(x)
		return x

class PPOMemory:
	def __init__(self, batch_size):
		self.states = []
		self.probs = []
		self.vals = []
		self.actions = []
		self.rewards = []
		self.dones = []

		self.batch_size = batch_size

	def generate_batches(self):
		n_states = len(self.states)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype=np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]

		return np.array(self.states),\
				np.array(self.actions),\
				np.array(self.probs),\
				np.array(self.vals),\
				np.array(self.rewards),\
				np.array(self.dones),\
				batches

	def store_memory(self, state, action, probs, vals, reward, done):
		self.states.append(state)
		self.actions.append(action)
		self.probs.append(probs)
		self.vals.append(vals)
		self.rewards.append(reward)
		self.dones.append(done)

	def clear_memory(self):
		self.states = []
		self.probs = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.vals = []

dis_width = 1700
dis_height = 956
NBMAXTOUR = 10
SIMULATION_TURNS = 4
SOLUTION_COUNT = 6
DX = 9.4117647058823529411764705882353
DY = 9.4142259414225941422594142259414
#DX = 1.0
#DY = 1.0
PI = 3.1415926535897932384626433832795

def distance(p1, p2):
	return math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))

def norme(p):
	return math.sqrt(p.x * p.x + p.y * p.y)

def det(p1, p2):
	return p1.x * p2.y - p1.y * p2.x 

def dot(p1, p2):
	return p1.x * p2.x + p1.y * p2.y

class Vector2(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

class Checkpoint:

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.radius = 50

class Sim(object):
	def __init__(self):
		self.pos = Vector2(0,0)
		self.speed = Vector2(0,0)
		self.angle = 0
		self.angletot = 0
		self.next_checkpoint = 1
		self.thrust = 5.0
		self.check_point = 1
		self.check_pass = 1
		self.direction = Vector2(1,1)

	def clone(self):
		new_sim = Sim()
		new_sim.pos.x = self.pos.x
		new_sim.pos.y = self.pos.y
		new_sim.speed.x = self.speed.x
		new_sim.speed.y = self.speed.y
		new_sim.angle = self.angle
		new_sim.angletot = self.angletot
		new_sim.next_checkpoint = self.next_checkpoint
		new_sim.thrust = self.thrust
		new_sim.check_point = self.check_point
		new_sim.check_pass = self.check_pass
		new_sim.direction.x = self.direction.x
		new_sim.direction.y = self.direction.y

		return new_sim

	def simulate(self):
		anglef = (self.angletot + self.angle + 360) % 360
		angle_rad = anglef * math.pi / 180.0
		direction = Vector2(math.cos(angle_rad) * float(self.thrust), math.sin(angle_rad) * float(self.thrust))
		self.speed = Vector2(self.speed.x + direction.x, self.speed.y + direction.y)
		self.pos.x += self.speed.x
		self.pos.y += self.speed.y

	def end_simulate(self):
		self.pos.x = int(self.pos.x)
		self.pos.y = int(self.pos.y)
		self.speed = Vector2(int(self.speed.x * 0.85), int(self.speed.y * 0.85))




class Solutionm:
	def __init__(self):
		self.moves1 = [Sim()] * 40  # Crée un tableau de taille 20 avec des valeurs None
		self.score = -2000000000

	def clone(self):
		new_solution = Solutionm()  # Crée une nouvelle instance de Solutionm
		new_solution.moves1 = [sim.clone() for sim in self.moves1]  # Copie le tableau moves1
		new_solution.score = self.score  # Copie la valeur de score
		return new_solution


class Simulation2(object):
	def __init__(self, nbs, d):
		self.NB_SOL = nbs
		self.DEPTH = d
		self.solution = []
		self.checkpoints = []
		self.podRadius = 400.0
		self.podRadiusSqr = self.podRadius * self.podRadius
		self.minImpulse = 120.0
		self.frictionFactor = 0.85
		self.MAXT = 200.0
		self.MAXTF = 200.0
		self.MINT = 0
		self.MAXA = 18
		self.MINA = -18
		self.state_chaser = 0
		self.state_chaser2 = 0
		self.reward = [0]*37
		self.rewardt = [0] * 5
		self.rewarda = [0] * 185

		for i in range(nbs):
			sol = Solutionm()
			for j in range(d):
				rng = random.Random()
				sm = Sim()
				angle = rng.randint(self.MINA, self.MAXA)
				sm.angle = min(max(angle, -18), 18)
				thrust = rng.randint(self.MINT, self.MAXT)
				sm.thrust = thrust#min(max(thrust, 0), 3)
				
				sol.moves1[j] = sm

			sol.score = -2000000000
			self.solution.append(sol)

		
	def Trier(self):
		self.solution.sort(key=lambda x: x.score, reverse=True)


	def mutate2(self, ind):
		dthrust = lambda: random.randint(0, 1)
		dangle = lambda: random.randint(0, 2)
		dcross = lambda: random.randint(0, 2)

		tangle = [-18, 0, 18]
		tthrust = [0, 200]

		sol = self.solution[ind].clone()

		for i in range(self.DEPTH):
			num = dcross()
			if num == 0:
				sol.moves1[i].angle = tangle[dangle()]
				sol.moves1[i].angle = max(-18, min(sol.moves1[i].angle, 18))
			
			elif num == 1:
				sol.moves1[i].thrust = tthrust[dthrust()]
				#sol.moves1[i].thrust = max(0, min(sol.moves1[i].thrust, 3))
			
			else:
				sol.moves1[i].angle = tangle[dangle()]
				sol.moves1[i].angle = max(-18, min(sol.moves1[i].angle, 18))
		

				sol.moves1[i].thrust = tthrust[dthrust()]
				#sol.moves1[i].thrust = max(0, min(sol.moves1[i].thrust, 3))
			

		return sol


	def play(self, p, turn, time):
		start_time = datetime.datetime.now()
		maxt = -1

		def get_time():
			global maxt
			stop_time = datetime.datetime.now()
			duration = (stop_time - start_time).total_seconds() * 1000  # Convertir en millisecondes
			maxt = int(duration)
			return duration <= time

		#dthrust = lambda: random.randint(self.MINT, self.MAXT)
		#dangle = lambda: random.randint(self.MINA, self.MAXA)
		dsol = lambda: random.randint(0, self.NB_SOL - 1)
		dgene = lambda: random.uniform(0, 1.0)

		dthrust = lambda: random.randint(0, 1)
		dangle = lambda: random.randint(0, 2)

		tangle = [-18, 0, 18]
		tthrust = [0, 200]


		if turn > 0:
			for n in range(self.NB_SOL):
				for d in range(self.DEPTH - 1):
					self.solution[n].moves1[d] = self.solution[n].moves1[d + 1].clone()
					self.solution[n].score = -2000000000

				i = self.DEPTH - 1
				self.solution[n].moves1[i].angle = tangle[dangle()]
				self.solution[n].moves1[i].angle = max(-18, min(self.solution[n].moves1[i].angle, 18))
		
				self.solution[n].moves1[i].thrust = tthrust[dthrust()]
				#self.solution[n].moves1[i].thrust = max(0, min(self.solution[n].moves1[i].thrust, 3))
			

		nb_turn = 0

		self.reward = [0]*37
		self.rewardt = [0] * 5
		self.rewarda = [0] * 185

		while get_time():
			score_chaser = 0

			for ind in range(self.NB_SOL):
				solret = self.mutate2(ind)

				pod1 = Sim()
				pod1.angletot = p.angletot;
				pod1.speed.x = p.speed.x;
				pod1.speed.y = p.speed.y;
				pod1.pos.x = p.pos.x
				pod1.pos.y = p.pos.y
				pod1.score = -2000000000;
				pod1.check_point = p.check_point;
				pod1.check_pass = p.check_pass;
				
				for i in range(self.DEPTH):
								
					pod1.angle = solret.moves1[i].angle
					pod1.thrust = solret.moves1[i].thrust
					#solret.moves1[i].thrust = 4
					#print(pod1.angle)
					#print(pod1.thrust)
					pod1.simulate()
					pod1.angletot = (int(pod1.angletot + pod1.angle + 360) % 360)
					pod1.end_simulate()

					p1dist = distance(pod1.pos, self.checkpoints[pod1.check_point])

					if p1dist <= 600:
						pod1.check_pass += 1
						pod1.check_point = (pod1.check_point + 1) % len(self.checkpoints)



				p1dist = distance(pod1.pos, self.checkpoints[pod1.check_point])
				score = (50000 * pod1.check_pass - p1dist)
				#self.reward[int(solret.moves1[0].angle) + 18] = score / 50000.0
				#self.rewardt[int(solret.moves1[0].thrust)] = score
				
				solret.score = score 

				if solret.score > self.solution[self.NB_SOL - 1].score:
					self.solution[self.NB_SOL - 1] = solret.clone()
					self.Trier()

					#for i in range(self.NB_SOL):
					#	print(str(i) +  " " + str(self.solution[i].score))


				nb_turn += 1

		num = int(self.solution[0].moves1[0].angle) + 18
		ind =  int(self.solution[0].moves1[0].thrust/50.0) * 37 + num
		self.rewarda[ind] = 1000000
		#print(nb_turn)
		#print(self.solution[0].score)

	def playRL(self, p, turn, time):
		start_time = datetime.datetime.now()
		maxt = -1

		def get_time():
			global maxt
			stop_time = datetime.datetime.now()
			duration = (stop_time - start_time).total_seconds() * 1000  # Convertir en millisecondes
			maxt = int(duration)
			return duration <= time

		dthrust = lambda: random.randint(self.MINT, self.MAXT)
		dangle = lambda: random.randint(self.MINA, self.MAXA)
		dsol = lambda: random.randint(0, self.NB_SOL - 1)
		dgene = lambda: random.uniform(0, 1.0)
				
		nb_turn = 0
				
		score_chaser = 0

		self.solution[0].score = -2000000000

		while get_time():
			pass

			
		

		
				
		index = 0;
		for ang in range(-18, 19):

			solret = Solutionm()

			pod1 = Sim()
			pod1.angletot = p.angletot;
			pod1.speed.x = p.speed.x;
			pod1.speed.y = p.speed.y;
			pod1.pos.x = p.pos.x
			pod1.pos.y = p.pos.y
			pod1.score = -2000000000;
			pod1.check_point = p.check_point;
			pod1.check_pass = p.check_pass;
			
			solret.moves1[0].angle = ang
			solret.moves1[0].thrust = dthrust()
			pod1.angle = ang
			pod1.thrust = 1
			#print(pod1.angle)
			#print(pod1.thrust)
			pod1.simulate()
			pod1.angletot = (int(pod1.angletot + pod1.angle + 360) % 360)
			pod1.end_simulate()

			p1dist = distance(pod1.pos, self.checkpoints[pod1.check_point])

			if p1dist <= 50:
				pod1.check_pass += 1
				pod1.check_point = (pod1.check_point + 1) % len(self.checkpoints)



			p1dist = distance(pod1.pos, self.checkpoints[pod1.check_point])
			score = 2000 * pod1.check_pass - p1dist
			solret.score = score + solret.moves1[0].thrust
				
			if solret.score > self.solution[0].score:
				self.solution[0]  =solret


			nb_turn += 1


def step(tensordict):

	#_thrust = (indt // 37)*50 
	#_angle = (indt % 37)-18

	sorties = [
		(-18, 0),   # sortie 1
		(0, 0),     # sortie 2
		(18, 0),    # sortie 3
		(-18, 200), # sortie 4
		(0, 200),   # sortie 5
		(18, 200)   # sortie 6
	]

	indt = tensordict["action"]
	sim = tensordict["pod"].sim
	simulation["pod"].simulation
		
	_thrust = sorties[indt][1]
	_angle = sorties[indt][0]
		
	anglef = int(sim.angletot + _angle + 360) % 360
	angleRad = anglef * PI / 180.0
	thrust = _thrust
	sim.thrust = thrust
	sim.angle = _angle
	#print(str(_thrust) + " " + str(_angle))
	dir_x = math.cos(angleRad) * float(thrust)
	dir_y = math.sin(angleRad) * float(thrust)
	sim.direction.x = math.cos(angleRad) * 1000
	sim.direction.y = math.sin(angleRad) * 1000
	sim.speed.x += dir_x
	sim.speed.y += dir_y
	sim.pos.x = round(sim.pos.x + sim.speed.x)
	sim.pos.y = round(sim.pos.y + sim.speed.y)
	sim.speed = Vector2(math.trunc(sim.speed.x * 0.85), math.trunc(sim.speed.y * 0.85))

	sim.angletot = int(sim.angletot +_angle + 360) % 360

	checkpoint = simulation.checkpoints

	#next_state
	next_check = (sim.check_point + 1) % len(checkpoint)
	last_check = (sim.check_point - 1 + len(checkpoint)) % len(checkpoint)
	x1, y1 = sim.pos.x, sim.pos.y
	x2, y2 = checkpoint[sim.check_point].x, checkpoint[sim.check_point].y
	x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
	angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
	angle = angle * 180.0 / math.pi
	angle = (angle + 180.0) % 360.0
	if angle < 0.0:
		angle += 360.0
	angle -= 180.0
	
	anglech = math.atan2(y2 - y1, x2 - x1)
	anglech = anglech * 180.0 / math.pi
	anglech = (anglech - sim.angletot + 540) % 360 - 180

	col = ((sim.speed.x * (checkpoint[sim.check_point].x - sim.pos.x) + sim.speed.y * (checkpoint[sim.check_point].y - sim.pos.y)) /
			(math.sqrt(sim.speed.x * sim.speed.x + sim.speed.y * sim.speed.y) *
			math.sqrt((checkpoint[sim.check_point].x - sim.pos.x) * (checkpoint[sim.check_point].x - sim.pos.x) +
						(checkpoint[sim.check_point].y - sim.pos.y) * (checkpoint[sim.check_point].y - sim.pos.y)) + 0.000001))
	
	dist_check = distance(checkpoint[sim.check_point], sim.pos)
	ndist_check = distance(checkpoint[sim.check_point], checkpoint[last_check])
		
				
	speed = norme(sim.speed)
	print("speed " + str(speed))

	a1 = (angle + 180) / 360.0
	a2 = (anglech + 180) / 360.0
	a3 = (col + 1.0) / 2.0
	a4 = (200000 - dist_check) / 200000.0
	a5 = speed / 10000.0
	a6 = sim.angletot / 360.0
	a7 = float(sim.thrust) / simulation.MAXTF
	a8 = float(sim.angle + 18.0) / 36.0 

	next_state = [a1, a2, a3, a4, a5, a6, a7, a8]

	reward = 1.0
		
	return next_state, reward


class SuperPod(gym.Env):

	def __init__(self):
		super(SuperPod, self).__init__()
		self.state_size = 8
		self.action_size = 185
		#self.memory = deque()
		self.temp_mem = deque()
		self.gamma = 0.99    # Discount rate
		self.epsilon = 0.9   # Exploration rate
		self.epsilon_min = 0.1
		self.epsilon_decay = 1000
		self.learning_rate = 0.001
		self.tau = 0.005

		self.policy_clip = 0.2
		self.n_epochs = 10
		self.gae_lambda = 0.95

		self.min_error = 2000000.0
		self.act_error = 0.0
		self.nb_rewards = -10000;
		self.act_rewards = 0.0;
		self.max_rewards_episode = -100000
		self.ind_episode = 0
		self.reward_episode = 0.0
		self.max_episode = 2
		self.act_reward_episode = 0
		self.act_dist = 0
		self.max_dist = 0
		self.plot_tloss = []
		self.plot_mloss = []

		self.nb_reach = 100
		self.action_space = spaces.Discrete(6)
		self.observation_space = spaces.Box(low=0, high=1.0, shape=(8,), dtype=np.float32)
		is_fork = multiprocessing.get_start_method() == "fork"
		self.device = (
			torch.device(0)
			if torch.cuda.is_available() and not is_fork
			else torch.device("cpu")
		)

				

		self.maps = [
			[Checkpoint(12460, 1350), Checkpoint(10540, 5980), Checkpoint(3580, 5180), Checkpoint(13580, 7600)],
			[Checkpoint(3600, 5280), Checkpoint(13840, 5080), Checkpoint(10680, 2280), Checkpoint(8700, 7460), Checkpoint(7200, 2160)],
			[Checkpoint(4560, 2180), Checkpoint(7350, 4940), Checkpoint(3320, 7230), Checkpoint(14580, 7700), Checkpoint(10560, 5060), Checkpoint(13100, 2320)],
			[Checkpoint(5010, 5260), Checkpoint(11480, 6080), Checkpoint(9100, 1840)],
			[Checkpoint(14660, 1410), Checkpoint(3450, 7220), Checkpoint(9420, 7240), Checkpoint(5970, 4240)],
			[Checkpoint(3640, 4420), Checkpoint(8000, 7900), Checkpoint(13300, 5540), Checkpoint(9560, 1400)],
			[Checkpoint(4100, 7420), Checkpoint(13500, 2340), Checkpoint(12940, 7220), Checkpoint(5640, 2580)],
			[Checkpoint(14520, 7780), Checkpoint(6320, 4290), Checkpoint(7800, 860), Checkpoint(7660, 5970), Checkpoint(3140, 7540), Checkpoint(9520, 4380)],
			[Checkpoint(10040, 5970), Checkpoint(13920, 1940), Checkpoint(8020, 3260), Checkpoint(2670, 7020)],
			[Checkpoint(7500, 6940), Checkpoint(6000, 5360), Checkpoint(11300, 2820)],
			[Checkpoint(4060, 4660), Checkpoint(13040, 1900), Checkpoint(6560, 7840), Checkpoint(7480, 1360), Checkpoint(12700, 7100)],
			[Checkpoint(3020, 5190), Checkpoint(6280, 7760), Checkpoint(14100, 7760), Checkpoint(13880, 1220), Checkpoint(10240, 4920), Checkpoint(6100, 2200)],
			[Checkpoint(10323, 3366), Checkpoint(11203, 5425), Checkpoint(7259, 6656), Checkpoint(5425, 2838)]
		]

		checkpoint = self.maps[random.randint(0, 12)]
		check = checkpoint[0]

		self.simulation = Simulation2(3, 7)
		self.sim = Sim()
		self.sim.pos.x = check.x
		self.sim.pos.y = check.y
		self.sim.speed = Vector2(0,1)
		self.sim.angletot = 0
		self.simulation.checkpoints = checkpoint
		self.sim.check_pass = 1
		self.sim.check_point = 1
		self.sim.thrust = self.simulation.MAXT
		self.sim.angle = 0;
		self.nn = 0#NeuralNetwork([7, 37, 37], 0.1)
		self.mangle = {-18:0, 0:1, 18:2}
		self.mthrust = {0:0, 200:1}
		
		self.nnt = 0#NeuralNetwork([7, 37, 4], 0.1)
		
		self.mon_reseau = MonReseau()
		# Charger les poids sauvegardés à partir du fichier
		#self.mon_reseau.load_state_dict(torch.load("bestnn.py.pth"))
		#self.mon_reseau.load_state_dict(torch.load("bestweight_nn.py.pth"))
		#self.mon_reseau.load_state_dict(torch.load("bestreward_nn.py.pth"))
		#self.mon_reseau.load_state_dict(torch.load("bestmaxrewardepisode_nn.py.pth"))
		#self.mon_reseau.load_state_dict(torch.load("bestnndist.py.pth"))
		
		self.criteres = nn.SmoothL1Loss() #nn.MSELoss(reduction='mean')
		self.optimiseur = optim.Adam(self.mon_reseau.parameters(), lr=0.0001)
		self.target_reseau = MonReseau()
		self.target_reseau.load_state_dict(self.mon_reseau.state_dict())

		self.value_reseau = MonReseauV()
		self.optimiseur_v = optim.Adam(self.value_reseau.parameters(), lr=0.0001)

		self.memory = PPOMemory(32)

		self.pass_check = 0

	def seed(self, seed=None):
		self.np_random, seed= seeding.np_random(seed)
					

	def convolution1D(self, input_data, kernel):
		input_size = len(input_data)
		kernel_size = len(kernel)
		output_size = input_size - kernel_size + 1
		output = [0.0] * output_size

		# Appliquer la convolution
		for i in range(output_size):
			for j in range(kernel_size):
				output[i] += input_data[i + j] * kernel[j]

		return output

	def normalizeInput(self, input):
		# Calculate mean
		mean = sum(input) / len(input)

		# Calculate standard deviation
		stddev = math.sqrt(sum((val - mean) ** 2 for val in input) / len(input))

		# Normalize input data
		for i in range(len(input)):
			input[i] = (input[i] - mean) / stddev



	def write_features_to_file(self, filename, features):
		with open(filename, 'a') as file:
			for feature_set in features:
				line = ' '.join(map(str, feature_set)) + '\n'
				file.write(line)

	def write_rewards_to_file(self, filename, rewards):
		with open(filename, 'a') as file:
			line = ''
			for feature_set in rewards:
				line += str(feature_set) + ' '
			line += '\n'
			file.write(line)

		
	def calculate_features(self, checkpoint):
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		last_check = (self.sim.check_point - 1 + len(checkpoint)) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		ndist_check = distance(checkpoint[self.sim.check_point], checkpoint[last_check])
		
				
		speed = norme(self.sim.speed)

		a1 = (angle + 180) / 360.0
		a2 = (anglech + 180) / 360.0
		a3 = (col + 1.0) / 2.0
		a4 = (200000 - dist_check) / 200000.0
		a5 = speed / 2000.0
		a6 = self.sim.angletot / 360.0
		a7 = float(self.sim.thrust) / self.simulation.MAXTF
		a8 = float(self.sim.angle + 18.0) / 36.0 
				

		self.write_features_to_file("datasett.txt", [(a1, a2, a3, a4, a5, a6, a7, a8)])
		



	def Move(self, turn, time):

		#self.calculate_features(self.simulation.checkpoints)

		self.simulation.play(self.sim, turn, time)

		
		maxl = -2000
		ind = -1
		for i in range(0, 37):
			if self.simulation.reward[i] > maxl:
				maxl = self.simulation.reward[i]
				ind = i

		rew = [0] * 37	
		rew[ind] = 1.0 
		
	
		#self.write_rewards_to_file("rewards.txt",rew)

		maxl = -2000
		ind = -1
		for i in range(0, 4):
			if self.simulation.rewardt[i] > maxl:
				maxl = self.simulation.rewardt[i]
				ind = i

		rewt = [0] * 4	
		rewt[ind] = 1.0 
	
		#self.write_rewards_to_file("rewardsth.txt",rewt)


		maxl = -2000
		ind = -1
		for i in range(0, 185):
			if self.simulation.rewarda[i] > maxl:
				maxl = self.simulation.rewarda[i]
				ind = i

		rewa = [0] * 185	
		rewa[ind] = 1.0 

		self.write_rewards_to_file("rewardat.txt",rewa)


		solm = self.simulation.solution[0]
				
		self.write_features_to_file("datasett.txt", [(ind, 0)])

		anglef = int(self.sim.angletot + solm.moves1[0].angle + 360) % 360
		angleRad = anglef * PI / 180.0
		thrust = solm.moves1[0].thrust
		self.sim.thrust = thrust
		self.sim.angle = solm.moves1[0].angle
		#print(str(thrust) + " " + str(solm.moves1[0].angle))
		dir_x = math.cos(angleRad) * float(thrust)
		dir_y = math.sin(angleRad) * float(thrust)
		self.sim.direction.x = math.cos(angleRad) * 1000
		self.sim.direction.y = math.sin(angleRad) * 1000
		self.sim.speed.x += dir_x
		self.sim.speed.y += dir_y
		self.sim.pos.x = int(self.sim.pos.x + self.sim.speed.x)
		self.sim.pos.y = int(self.sim.pos.y + self.sim.speed.y)
		self.sim.speed = Vector2(int(self.sim.speed.x * 0.85), int(self.sim.speed.y * 0.85))

		self.sim.angletot = int(self.sim.angletot + solm.moves1[0].angle + 360) % 360
	

	def calculate_features2(self, checkpoint):
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		next_check2 = (self.sim.check_point + 2) % len(checkpoint)
		last_check = (self.sim.check_point - 1 + len(checkpoint)) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		ndist_check = distance(checkpoint[self.sim.check_point], checkpoint[last_check])
		
				
		speed = norme(self.sim.speed)


		anglef = int(self.sim.angletot + self.sim.angle + 360) % 360
		angleRad = anglef * PI / 180.0
		dir_x = math.cos(angleRad) * 1000.0
		dir_y = math.sin(angleRad) * 1000.0

		px = Vector2(dir_x, 0.0);
		py = Vector2(0.0, dir_y);

		dirCPx = checkpoint[self.sim.check_point].x - self.sim.pos.x;
		dirCPy = checkpoint[self.sim.check_point].y - self.sim.pos.y;

		dirCP2x = checkpoint[next_check].x - self.sim.pos.x;
		dirCP2y = checkpoint[next_check].y - self.sim.pos.y;

		dirCP3x = checkpoint[next_check2].x - self.sim.pos.x;
		dirCP3y = checkpoint[next_check2].y - self.sim.pos.y;

		a1 = dot(px, self.sim.speed) / 20000000.0
		a2 = dot(py, self.sim.speed) / 20000000.0
		a3 = dot(px, Vector2(dirCPx, dirCPy)) / 20000000.0
		a4 = dot(py, Vector2(dirCPx, dirCPy)) / 20000000.0
		a5 = dot(px, Vector2(dirCP2x, dirCP2y))/ 20000000.0
		a6 = dot(py, Vector2(dirCP2x, dirCP2y))/ 20000000.0
		a7 = dot(px, Vector2(dirCP3x, dirCP3y))/ 20000000.0
		a8 = dot(py, Vector2(dirCP3x, dirCP3y))/ 20000000.0
						

		self.write_features_to_file("dataset2.txt", [(a1, a2, a3, a4, a5, a6, a7, a8)])

	def Move2(self, turn, time):

		self.calculate_features2(self.simulation.checkpoints)

		self.simulation.play(self.sim, turn, time)

		
		maxl = -2000
		ind = -1
		for i in range(0, 37):
			if self.simulation.reward[i] > maxl:
				maxl = self.simulation.reward[i]
				ind = i

		rew = [0] * 37	
		rew[ind] = 1.0 
		
	
		self.write_rewards_to_file("rewards2.txt",rew)

		maxl = -2000
		ind = -1
		for i in range(0, 4):
			if self.simulation.rewardt[i] > maxl:
				maxl = self.simulation.rewardt[i]
				ind = i

		rewt = [0] * 4	
		rewt[ind] = 1.0 
	
		self.write_rewards_to_file("rewardsth2.txt",rewt)


		maxl = -2000
		ind = -1
		for i in range(0, 185):
			if self.simulation.rewarda[i] > maxl:
				maxl = self.simulation.rewarda[i]
				ind = i

		rewa = [0] * 185	
		rewa[ind] = 1.0 

		self.write_rewards_to_file("rewarda2.txt",rewa)


		solm = self.simulation.solution[0]
				
		self.write_features_to_file("dataset2.txt", [(ind, 0)])

		anglef = int(self.sim.angletot + solm.moves1[0].angle + 360) % 360
		angleRad = anglef * PI / 180.0
		thrust = solm.moves1[0].thrust
		self.sim.thrust = thrust
		self.sim.angle = solm.moves1[0].angle
		print(str(thrust) + " " + str(solm.moves1[0].angle))
		dir_x = math.cos(angleRad) * float(thrust)
		dir_y = math.sin(angleRad) * float(thrust)
		self.sim.direction.x = math.cos(angleRad) * 100
		self.sim.direction.y = math.sin(angleRad) * 100
		self.sim.speed.x += dir_x
		self.sim.speed.y += dir_y
		self.sim.pos.x = int(self.sim.pos.x + self.sim.speed.x)
		self.sim.pos.y = int(self.sim.pos.y + self.sim.speed.y)
		self.sim.speed = Vector2(int(self.sim.speed.x * 0.85), int(self.sim.speed.y * 0.85))

		self.sim.angletot = int(self.sim.angletot + solm.moves1[0].angle + 360) % 360

	def scale_output_value(self, output_value, new_min, new_max):
		scaled_value = output_value * (new_max - new_min) + new_min
		return scaled_value

	def softmax_rlp(self, tab):
		max_x = np.max(tab)  # Trouver la valeur maximale de x
		exp_x = np.exp(tab - max_x)  # Calculer les exponentielles des éléments du vecteur
		sum_exp_x = np.sum(exp_x)  # Calculer la somme des exponentielles

		# Calculer le softmax
		softmax_values = exp_x / sum_exp_x
		sum_softmax = np.sum(softmax_values)  # Calculer la somme des valeurs softmax
		softmax_values /= sum_softmax 
		return softmax_values

	def softmax_rl(self):
		max_x = np.max(self.nn.network[-1])  # Trouver la valeur maximale de x
		exp_x = np.exp(self.nn.network[-1] - max_x)  # Calculer les exponentielles des éléments du vecteur
		sum_exp_x = np.sum(exp_x)  # Calculer la somme des exponentielles

		# Calculer le softmax
		softmax_values = exp_x / sum_exp_x
		sum_softmax = np.sum(softmax_values)  # Calculer la somme des valeurs softmax
		softmax_values /= sum_softmax 
		return softmax_values

	def softmax_rlt(self):
		max_x = np.max(self.nnt.network[-1])  # Trouver la valeur maximale de x
		exp_x = np.exp(self.nnt.network[-1] - max_x)  # Calculer les exponentielles des éléments du vecteur
		sum_exp_x = np.sum(exp_x)  # Calculer la somme des exponentielles

		# Calculer le softmax
		softmax_values = exp_x / sum_exp_x
		sum_softmax = np.sum(softmax_values)  # Calculer la somme des valeurs softmax
		softmax_values /= sum_softmax 
		return softmax_values

	def MoveNN(self, turn, time):
		start_time = datetime.datetime.now()
		def get_time():
			stop_time = datetime.datetime.now()
			duration = (stop_time - start_time).total_seconds() * 1000  # Convertir en millisecondes
			return duration <= time

		while(get_time()):
			pass

		checkpoint = self.simulation.checkpoints
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		speed = norme(self.sim.speed)

		a1 = (angle + 180) / 360.0
		a2 = (anglech + 180) / 360.0
		a3 = (col + 1.0) / 2.0
		a4 = (2000 - dist_check) / 2000.0
		a5 = speed / 2000.0
		a6 = self.sim.angletot / 360.0
		a7 = float(self.sim.thrust) / self.simulation.MAXTF
		a8 = float(self.sim.angle + 18) / 36.0 

		input_datat = [a1, a2, a3, a4, a5, a6, a7]
		self.nnt.SetInput(input_datat)
		self.nnt.normalizeInput()
		self.nnt.PredictNN()
			
		softt = self.softmax_rlt();
		maxlt =-20000
		indt = -1
		for i in range(0, 4):
			if softt[i] > maxlt:
				maxlt = softt[i]
				indt = i

		_thrust = indt
		#print(_thrust)

		a7 = float(_thrust) / self.simulation.MAXTF

		input_data = [a1, a2, a3, a4, a5, a6, a7]
		kernel = [0.5, 0.5]
		#self.normalizeInput(input_data)
		#self.nn.SetInput(self.convolution1D(input_data, kernel))
		self.nn.SetInput(input_data)
		self.nn.normalizeInput()
		self.nn.PredictNNT()

		
		soft = self.softmax_rl();
		maxl =-20000
		ind = -1
		for i in range(0, 37):
			#print(soft[i])
			if soft[i] > maxl:
				maxl = soft[i]
				ind = i
		
					
		_angle = ind - 18.0


		

		#_angle = self.scale_output_value(self.nn.network[-1][0], -18, 18)
		#_thrust = self.scale_output_value(self.nn.network[-1][1], 0, 250)
		#_thrust *= 2
		#_angle *= 2
		#print(speed)
		#print(str(_angle) + " " + str(_thrust))

		anglef = int(self.sim.angletot + _angle + 360) % 360
		angleRad = anglef * PI / 180.0
		thrust = _thrust
		self.sim.thrust = thrust
		self.sim.angle = _angle
		#print(str(thrust) + " " + str(solm.moves1[0].angle))
		dir_x = math.cos(angleRad) * float(thrust)
		dir_y = math.sin(angleRad) * float(thrust)
		self.sim.direction.x = math.cos(angleRad) * 100
		self.sim.direction.y = math.sin(angleRad) * 100
		self.sim.speed.x += dir_x
		self.sim.speed.y += dir_y
		self.sim.pos.x = int(self.sim.pos.x + self.sim.speed.x)
		self.sim.pos.y = int(self.sim.pos.y + self.sim.speed.y)
		self.sim.speed = Vector2(int(self.sim.speed.x * 0.85), int(self.sim.speed.y * 0.85))

		self.sim.angletot = int(self.sim.angletot + _angle + 360) % 360

	def softmax_rla(self):
		max_x = np.max(self.nna.network[-1])  # Trouver la valeur maximale de x
		exp_x = np.exp(self.nna.network[-1] - max_x)  # Calculer les exponentielles des éléments du vecteur
		sum_exp_x = np.sum(exp_x)  # Calculer la somme des exponentielles

		# Calculer le softmax
		softmax_values = exp_x / sum_exp_x
		sum_softmax = np.sum(softmax_values)  # Calculer la somme des valeurs softmax
		softmax_values /= sum_softmax 
		return softmax_values

	def MoveNN2(self, turn, time):
		start_time = datetime.datetime.now()
		def get_time():
			stop_time = datetime.datetime.now()
			duration = (stop_time - start_time).total_seconds() * 1000  # Convertir en millisecondes
			return duration <= time

		while(get_time()):
			pass

		checkpoint = self.simulation.checkpoints
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		speed = norme(self.sim.speed)

		print("speed " + str(speed))

		a1 = (angle + 180) / 360.0
		a2 = (anglech + 180) / 360.0
		a3 = (col + 1.0) / 2.0
		a4 = (200000 - dist_check) / 200000.0
		a5 = speed / 10000.0
		a6 = self.sim.angletot / 360.0
		a7 = float(self.sim.thrust) / self.simulation.MAXTF
		a8 = float(self.sim.angle + 18) / 36.0 

		input_datat = [a1, a2, a3, a4, a5, a6, a7, a8]
		donnees = torch.tensor(input_datat, dtype=torch.float32)
		predictions = self.mon_reseau(donnees)
		action = predictions.sample()
		indt = torch.squeeze(action).item()
		# Obtenir l'indice de la valeur maximale dans predictions
		#max_val, indt = torch.max(predictions, dim=0)
		#print("----------------------", indt.item())
		# Calculer le thrust et l'angle
		#_thrust = (indt.item() // 37) *50
		#_angle = (indt.item() % 37) - 18

		sorties = [
			(-18, 0),   # sortie 1
			(0, 0),     # sortie 2
			(18, 0),    # sortie 3
			(-18, 200), # sortie 4
			(0, 200),   # sortie 5
			(18, 200)   # sortie 6
		]

		
		_thrust = sorties[indt][1]
		_angle = sorties[indt][0]

		
		#_thrust = indt % 5
		#_angle = (indt // 5) - 18


		

		#_angle = self.scale_output_value(self.nn.network[-1][0], -18, 18)
		#_thrust = self.scale_output_value(self.nn.network[-1][1], 0, 250)
		#_thrust *= 2
		#_angle *= 2
		#print(speed)
		#print(str(_angle) + " " + str(_thrust))

		anglef = int(self.sim.angletot + _angle + 360) % 360
		angleRad = anglef * PI / 180.0
		thrust = _thrust
		self.sim.thrust = thrust
		self.sim.angle = _angle
		#print(str(thrust) + " " + str(solm.moves1[0].angle))
		dir_x = math.cos(angleRad) * float(thrust)
		dir_y = math.sin(angleRad) * float(thrust)
		self.sim.direction.x = math.cos(angleRad) * 1000
		self.sim.direction.y = math.sin(angleRad) * 1000
		self.sim.speed.x += dir_x
		self.sim.speed.y += dir_y
		self.sim.pos.x = round(self.sim.pos.x + self.sim.speed.x)
		self.sim.pos.y = round(self.sim.pos.y + self.sim.speed.y)
		#print(str(self.sim.pos.x) + " " + str(self.sim.pos.y))
		print(self.sim.pos.x, " " , self.sim.pos.y)
		self.sim.speed = Vector2(math.trunc(self.sim.speed.x * 0.85), math.trunc(self.sim.speed.y * 0.85))

		self.sim.angletot = int(self.sim.angletot + _angle + 360) % 360


	def MoveModel(self, turn, time, model):
		start_time = datetime.datetime.now()
		def get_time():
			stop_time = datetime.datetime.now()
			duration = (stop_time - start_time).total_seconds() * 1000  # Convertir en millisecondes
			return duration <= time

		while(get_time()):
			pass

		checkpoint = self.simulation.checkpoints
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		speed = norme(self.sim.speed)

		print("speed " + str(speed))

		a1 = (angle + 180) / 360.0
		a2 = (anglech + 180) / 360.0
		a3 = (col + 1.0) / 2.0
		a4 = (200000 - dist_check) / 200000.0
		a5 = speed / 10000.0
		a6 = self.sim.angletot / 360.0
		a7 = float(self.sim.thrust) / self.simulation.MAXTF
		a8 = float(self.sim.angle + 18) / 36.0 

		input_datat = [a1, a2, a3, a4, a5, a6, a7, a8]
		input_datat = np.array(input_datat)
		indt, _states = model.predict(input_datat)

		 

		# Obtenir l'indice de la valeur maximale dans predictions
		#max_val, indt = torch.max(predictions, dim=0)
		#print("----------------------", indt.item())
		# Calculer le thrust et l'angle
		#_thrust = (indt.item() // 37) *50
		#_angle = (indt.item() % 37) - 18

		sorties = [
			(-18, 0),   # sortie 1
			(0, 0),     # sortie 2
			(18, 0),    # sortie 3
			(-18, 200), # sortie 4
			(0, 200),   # sortie 5
			(18, 200)   # sortie 6
		]

		
		_thrust = sorties[indt][1]
		_angle = sorties[indt][0]

		
		#_thrust = indt % 5
		#_angle = (indt // 5) - 18


		

		#_angle = self.scale_output_value(self.nn.network[-1][0], -18, 18)
		#_thrust = self.scale_output_value(self.nn.network[-1][1], 0, 250)
		#_thrust *= 2
		#_angle *= 2
		#print(speed)
		#print(str(_angle) + " " + str(_thrust))

		anglef = int(self.sim.angletot + _angle + 360) % 360
		angleRad = anglef * PI / 180.0
		thrust = _thrust
		self.sim.thrust = thrust
		self.sim.angle = _angle
		#print(str(thrust) + " " + str(solm.moves1[0].angle))
		dir_x = math.cos(angleRad) * float(thrust)
		dir_y = math.sin(angleRad) * float(thrust)
		self.sim.direction.x = math.cos(angleRad) * 1000
		self.sim.direction.y = math.sin(angleRad) * 1000
		self.sim.speed.x += dir_x
		self.sim.speed.y += dir_y
		self.sim.pos.x = round(self.sim.pos.x + self.sim.speed.x)
		self.sim.pos.y = round(self.sim.pos.y + self.sim.speed.y)
		#print(str(self.sim.pos.x) + " " + str(self.sim.pos.y))
		print(self.sim.pos.x, " " , self.sim.pos.y)
		self.sim.speed = Vector2(math.trunc(self.sim.speed.x * 0.85), math.trunc(self.sim.speed.y * 0.85))

		self.sim.angletot = int(self.sim.angletot + _angle + 360) % 360


	def MoveNNNew(self, turn, time):
		start_time = datetime.datetime.now()
		def get_time():
			stop_time = datetime.datetime.now()
			duration = (stop_time - start_time).total_seconds() * 1000  # Convertir en millisecondes
			return duration <= time

		while(get_time()):
			pass

		checkpoint = self.simulation.checkpoints
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		next_check2 = (self.sim.check_point + 2) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		speed = norme(self.sim.speed)

		#New
		anglef = int(self.sim.angletot + self.sim.angle + 360) % 360
		angleRad = anglef * PI / 180.0
		dir_x = math.cos(angleRad) * 1000.0
		dir_y = math.sin(angleRad) * 1000.0

		px = Vector2(dir_x, 0.0);
		py = Vector2(0.0, dir_y);

		dirCPx = checkpoint[self.sim.check_point].x - self.sim.pos.x;
		dirCPy = checkpoint[self.sim.check_point].y - self.sim.pos.y;

		dirCP2x = checkpoint[next_check].x - self.sim.pos.x;
		dirCP2y = checkpoint[next_check].y - self.sim.pos.y;

		dirCP3x = checkpoint[next_check2].x - self.sim.pos.x;
		dirCP3y = checkpoint[next_check2].y - self.sim.pos.y;

		a1 = dot(px, self.sim.speed) / 20000000.0
		a2 = dot(py, self.sim.speed) / 20000000.0
		a3 = dot(px, Vector2(dirCPx, dirCPy))/ 20000000.0
		a4 = dot(py, Vector2(dirCPx, dirCPy))/ 20000000.0
		a5 = dot(px, Vector2(dirCP2x, dirCP2y))/ 20000000.0
		a6 = dot(py, Vector2(dirCP2x, dirCP2y))/ 20000000.0
		a7 = dot(px, Vector2(dirCP3x, dirCP3y))/ 20000000.0
		a8 = dot(py, Vector2(dirCP3x, dirCP3y))/ 20000000.0

		input_datat = [a1, a2, a3, a4, a5, a6, a7, a8]
		self.nna.SetInput(input_datat)
		self.nna.normalizeInput()
		self.nna.PredictNNRL()
			
		softt = self.softmax_rla();
		maxlt =-20000
		indt = -1
		for i in range(0, 185):
			if softt[i] > maxlt:
				maxlt = softt[i]
				indt = i

		_thrust = indt // 37
		_angle = (indt % 37)-18

		
		#_thrust = indt % 5
		#_angle = (indt // 5) - 18


		

		#_angle = self.scale_output_value(self.nn.network[-1][0], -18, 18)
		#_thrust = self.scale_output_value(self.nn.network[-1][1], 0, 250)
		#_thrust *= 2
		#_angle *= 2
		#print(speed)
		#print(str(_angle) + " " + str(_thrust))

		anglef = int(self.sim.angletot + _angle + 360) % 360
		angleRad = anglef * PI / 180.0
		thrust = _thrust
		self.sim.thrust = thrust
		self.sim.angle = _angle
		#print(str(thrust) + " " + str(solm.moves1[0].angle))
		dir_x = math.cos(angleRad) * float(thrust)
		dir_y = math.sin(angleRad) * float(thrust)
		self.sim.direction.x = math.cos(angleRad) * 100
		self.sim.direction.y = math.sin(angleRad) * 100
		self.sim.speed.x += dir_x
		self.sim.speed.y += dir_y
		self.sim.pos.x = int(self.sim.pos.x + self.sim.speed.x)
		self.sim.pos.y = int(self.sim.pos.y + self.sim.speed.y)
		print(self.sim.pos.x, " " , self.sim.pos.y)
		self.sim.speed = Vector2(int(self.sim.speed.x * 0.85), int(self.sim.speed.y * 0.85))

		self.sim.angletot = int(self.sim.angletot + _angle + 360) % 360

	def get_state(self, checkpoint):
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		last_check = (self.sim.check_point - 1 + len(checkpoint)) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		ndist_check = distance(checkpoint[self.sim.check_point], checkpoint[last_check])
		
				
		speed = norme(self.sim.speed)

		a1 = (angle + 180) / 360.0
		a2 = (anglech + 180) / 360.0
		a3 = (col + 1.0) / 2.0
		a4 = (200000 - dist_check) / 200000.0
		a5 = speed / 10000.0
		a6 = self.sim.angletot / 360.0
		a7 = float(self.sim.thrust) / self.simulation.MAXTF
		a8 = float(self.sim.angle + 18.0) / 36.0 

		return [a1, a2, a3, a4, a5, a6, a7, a8]

	
	def step(self, indt):

		#_thrust = (indt // 37)*50 
		#_angle = (indt % 37)-18

		sorties = [
			(-18, 0),   # sortie 1
			(0, 0),     # sortie 2
			(18, 0),    # sortie 3
			(-18, 200), # sortie 4
			(0, 200),   # sortie 5
			(18, 200)   # sortie 6
		]

		
		_thrust = sorties[indt][1]
		_angle = sorties[indt][0]
		
		anglef = int(self.sim.angletot + _angle + 360) % 360
		angleRad = anglef * PI / 180.0
		thrust = _thrust
		self.sim.thrust = thrust
		self.sim.angle = _angle
		#print(str(_thrust) + " " + str(_angle))
		dir_x = math.cos(angleRad) * float(thrust)
		dir_y = math.sin(angleRad) * float(thrust)
		self.sim.direction.x = math.cos(angleRad) * 1000
		self.sim.direction.y = math.sin(angleRad) * 1000
		self.sim.speed.x += dir_x
		self.sim.speed.y += dir_y
		self.sim.pos.x = round(self.sim.pos.x + self.sim.speed.x)
		self.sim.pos.y = round(self.sim.pos.y + self.sim.speed.y)
		self.sim.speed = Vector2(math.trunc(self.sim.speed.x * 0.85), math.trunc(self.sim.speed.y * 0.85))

		self.sim.angletot = int(self.sim.angletot +_angle + 360) % 360

		checkpoint = self.simulation.checkpoints

		#next_state
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		last_check = (self.sim.check_point - 1 + len(checkpoint)) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		ndist_check = distance(checkpoint[self.sim.check_point], checkpoint[last_check])
		
				
		speed = norme(self.sim.speed)
		#print("speed " + str(speed))

		a1 = (angle + 180) / 360.0
		a2 = (anglech + 180) / 360.0
		a3 = (col + 1.0) / 2.0
		a4 = (200000 - dist_check) / 200000.0
		a5 = speed / 10000.0
		a6 = self.sim.angletot / 360.0
		a7 = float(self.sim.thrust) / self.simulation.MAXTF
		a8 = float(self.sim.angle + 18.0) / 36.0 

		reward = a4 * 0.1
		done = 0
		self.nb_reach -= 1
		if self.nb_reach < 0:
			done = 1
			reward = -1
			self.nb_reach = 100

		d = distance(self.sim.pos, checkpoint[self.sim.check_point])
		#print("distance " + str(d))
		if(d <= 600):
			self.sim.check_point = (self.sim.check_point + 1) % len(checkpoint)
			self.sim.check_pass += 1
			self.pass_check = 1
			reward = 1.0
			self.nb_reach = 100
			
			if self.sim.check_point == 0:
				check = self.simulation.checkpoints[0]
				checkpoint = self.maps[random.randint(0, len(self.maps)-1)] #generate_checkpoints2(check, 16000, 9000, 2000, 5)
				self.simulation.checkpoints = checkpoint


		if d > 15000:
			reward = -1

		#if self.sim.pos.x < 0 or self.sim.pos.x > 16000 or self.sim.pos.y < 0 or self.sim.pos.y > 9000:
		#	reward = -1		


		info = {}

		next_state = [a1, a2, a3, a4, a5, a6, a7, a8]
		next_state = np.array(next_state)
		
		
		return next_state, reward, done, info

	def reset(self):
		check = self.simulation.checkpoints[0]

								
		self.sim.pos.x = check.x
		self.sim.pos.y = check.y
		self.sim.speed = Vector2(0,1)
		self.sim.check_point = 1 #(pod.sim.check_point + 1) % len(checkpoint)
		self.sim.check_pass = 1
		self.nb_reach = 100

		state = np.array(self.get_state(self.simulation.checkpoints))

		return state
		

	def render(self):
		pass

	
	def remember_tmp(self, state, action, reward, next_state, done):
		self.temp_mem.append((state, action, reward, next_state, done))

	def remember(self, state, action, probs, vals, reward, done):
		self.memory.store_memory(state, action, probs, vals, reward, done)

	def choose_action(self, observation, turn, times):

		speeds = norme(self.sim.speed)
		print("speed " , speeds)
		
		fl = random.random()

		eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1. * turn / self.epsilon_decay)

		print("epsilon ", fl, " ", eps_threshold)
		
		#if fl <= eps_threshold:
		self.simulation.play(self.sim, turn, times)
		ind = self.mthrust[self.simulation.solution[0].moves1[0].thrust] * 3 +  self.mangle[self.simulation.solution[0].moves1[0].angle]

		return ind, 1.0, 1.0

		with torch.no_grad():

			state = torch.tensor([observation], dtype=torch.float)

			dist = self.mon_reseau(state)
			value = self.value_reseau(state)
			action = dist.sample()

			#print("action " , action)
	
			probs = torch.squeeze(dist.log_prob(action)).item()
			action = torch.squeeze(action).item()
			value = torch.squeeze(value).item()

			print("CHOOSE HIMSELF ------------------------------------------*****************************************************" + str(action))
			#print("probs", probs)
			#print("value", value)

			return action, probs, value
		
	def act(self, state, turn, time):

		speeds = norme(self.sim.speed)
		print("speed " , speeds)
		
		fl = random.random()

		eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1. * turn / self.epsilon_decay)

		print("epsilon ", fl, " ", eps_threshold)
		
		if fl <= eps_threshold:
			self.simulation.play(self.sim, turn, time)
			#num = int(self.simulation.solution[0].moves1[0].angle) + 18
			#ind =  int(self.simulation.solution[0].moves1[0].thrust//50) * 37 + num
			ind = self.mthrust[self.simulation.solution[0].moves1[0].thrust] * 3 +  self.mangle[self.simulation.solution[0].moves1[0].angle]

			return ind
		

		"""
		checkpoint = self.simulation.checkpoints
		next_check = (self.sim.check_point + 1) % len(checkpoint)
		x1, y1 = self.sim.pos.x, self.sim.pos.y
		x2, y2 = checkpoint[self.sim.check_point].x, checkpoint[self.sim.check_point].y
		x3, y3 = checkpoint[next_check].x, checkpoint[next_check].y
	
		angle = math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
		angle = angle * 180.0 / math.pi
		angle = (angle + 180.0) % 360.0
		if angle < 0.0:
			angle += 360.0
		angle -= 180.0
	
		anglech = math.atan2(y2 - y1, x2 - x1)
		anglech = anglech * 180.0 / math.pi
		anglech = (anglech - self.sim.angletot + 540) % 360 - 180

		col = ((self.sim.speed.x * (checkpoint[self.sim.check_point].x - self.sim.pos.x) + self.sim.speed.y * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) /
			   (math.sqrt(self.sim.speed.x * self.sim.speed.x + self.sim.speed.y * self.sim.speed.y) *
				math.sqrt((checkpoint[self.sim.check_point].x - self.sim.pos.x) * (checkpoint[self.sim.check_point].x - self.sim.pos.x) +
						  (checkpoint[self.sim.check_point].y - self.sim.pos.y) * (checkpoint[self.sim.check_point].y - self.sim.pos.y)) + 0.000001))
	
		dist_check = distance(checkpoint[self.sim.check_point], self.sim.pos)
		speed = norme(self.sim.speed)

		a1 = (angle + 180) / 360.0
		a2 = (anglech + 180) / 360.0
		a3 = (col + 1.0) / 2.0
		a4 = (200000 - dist_check) / 200000.0
		a5 = speed / 2000.0
		a6 = self.sim.angletot / 360.0
		a7 = float(self.sim.thrust) / self.simulation.MAXTF
		a8 = float(self.sim.angle + 18) / 36.0 
		"""

		a1 = state[0]
		a2 = state[1]
		a3 = state[2]
		a4 = state[3]
		a5 = state[4]
		a6 = state[5]
		a7 = state[6]
		a8 = state[7]
				
		input_datat = [a1, a2, a3, a4, a5, a6, a7, a8]
		donnees = torch.tensor(input_datat, dtype=torch.float32)
		#mean = torch.mean(donnees, dim=0)
		#std = torch.std(donnees, dim=0)

		# Normaliser les données d'entrée
		#donnees_normalisees = (donnees - mean) / std
		with torch.no_grad():
			predictions = self.mon_reseau(donnees)
			
			# Obtenir l'indice de la valeur maximale dans predictions
			max_val, indt = torch.max(predictions, dim=0)
			
			print("CHOOSE HIMSELF ------------------------------------------*****************************************************" + str(indt.item()))
			return indt.item()

	
	def learn(self):

		LOSS = 0.0
		for _ in range(self.n_epochs):
			state_arr, action_arr, old_prob_arr, vals_arr,\
			reward_arr, dones_arr, batches = \
					self.memory.generate_batches()

			values = vals_arr
			advantage = np.zeros(len(reward_arr), dtype=np.float32)

			for t in range(len(reward_arr)-1):
				discount = 1
				a_t = 0
				for k in range(t, len(reward_arr)-1):
					a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
							(1-int(dones_arr[k])) - values[k])
					discount *= self.gamma*self.gae_lambda
				advantage[t] = a_t
			advantage = torch.tensor(advantage)

			values = torch.tensor(values)
			for batch in batches:
				states = torch.tensor(state_arr[batch], dtype=torch.float)
				old_probs = torch.tensor(old_prob_arr[batch])
				actions = torch.tensor(action_arr[batch])

				dist = self.mon_reseau(states)
				critic_value = self.value_reseau(states)

				critic_value = torch.squeeze(critic_value)

				new_probs = dist.log_prob(actions)
				prob_ratio = new_probs.exp() / old_probs.exp()
				#prob_ratio = (new_probs - old_probs).exp()
				weighted_probs = advantage[batch] * prob_ratio
				weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
						1+self.policy_clip)*advantage[batch]
				actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

				returns = advantage[batch] + values[batch]
				critic_loss = (returns-critic_value)**2
				critic_loss = critic_loss.mean()

				total_loss = actor_loss + 0.5*critic_loss
				LOSS += total_loss.item()
				self.optimiseur.zero_grad()
				self.optimiseur_v.zero_grad()
				total_loss.backward()
				self.optimiseur.step()
				self.optimiseur_v.step()

		self.memory.clear_memory()    

		self.act_error = LOSS
		if LOSS < self.min_error:
			self.min_error = LOSS
			

def generate_checkpoints2(check, width, height, min_distance, nb):
	checkpoints = [check]
	while True:
		x = random.randint(800, width-800)
		y = random.randint(800, height-800)
		min_distance = random.randint(800, min_distance)
		new_checkpoint = Checkpoint(x, y)
		if all(distance(new_checkpoint, checkpoint) >= min_distance for checkpoint in checkpoints):
			checkpoints.append(new_checkpoint)
			if len(checkpoints) >= nb:  # Sortir après avoir généré 10 checkpoints
				break
	return checkpoints

# Fonction pour déplacer le carré
def deplacer_carre():
	canvas.move(carre, 5, 0)  # Déplacer le carré de 5 pixels vers la droite
	fenetre.after(75, deplacer_carre)  # Appeler cette fonction toutes les 75 ms

def draw_car():
	global turn
	global state
	global next_state
	global reward 
	global batch_size 
	global nb_reach 
	global done
	global action 
	global value
	global prob
	global checkpoint
	global times
	global pod
	global maps
	global steps
	global max_check
	global info
	global model
			
	draw_checkpoint(pod.simulation.checkpoints, 63)

	if moveNN:
		#pod.MoveNN2(turn, 10)
		pod.MoveModel(turn, 10, model)
	else:
		#action = pod.act(state, steps, 10)
		action, prob, value = pod.choose_action(state, steps, 10)
		steps += 1
		next_state, reward, done, info = pod.step(action)
				
		if nb_reach < 0:
			pod.reset()
					


	x = pod.sim.pos.x / DX
	y = pod.sim.pos.y / DY

	canvas.delete("car")
	canvas.create_rectangle(x-25, y-25, x+25, y+25, fill="red", tags="car")

	if moveNN:
		d = distance(pod.sim.pos, pod.simulation.checkpoints[pod.sim.check_point])
		#print("distance " + str(d))
		if(d <= 600):
			pod.sim.check_point = (pod.sim.check_point + 1) % len(pod.simulation.checkpoints)
			pod.sim.check_pass += 1
			pod.pass_check = 1
			
						

	if not moveNN:
		
		#if pod.sim.pos.x < 0 or pod.sim.pos.x > 16000 or pod.sim.pos.y < 0 or pod.sim.pos.y > 9000:
		#	reward = -1		
			
		#if d > 20000:
		#	reward = -1
			
		#pod.replay_one(state, action, reward, next_state, done, batch_size)
		#pod.remember(state, action, prob, value, reward, done)
		#state = next_state
				
		#if (turn+1) % 100 == 0:
		#	pod.learn()
			
		label_error.config(text="MIN_ERROR = " + str(pod.min_error) + ", ACT_ERROR = " + str(pod.act_error) + ", MAX_REWARDS = " +  str(pod.nb_rewards) +
		", ACT_REWARDS = " +  str(pod.act_rewards) + ", ACT_REWARDS_EPISODE = " +  str(pod.act_reward_episode) + ", MAX_REWARDS_EPISODE = " +  str(pod.max_rewards_episode) +
		", ACT_DIST = " +  str(pod.act_dist) + ", ACT_REWARDS_EPISODE = " +  str(pod.max_dist) + ", TURN = " + str(turn))

		"""
		if turn % 10 == 0:
			if pod.epsilon > pod.epsilon_min:
				pod.epsilon *= pod.epsilon_decay
			else:
				pod.epsilon = 1.0
		"""

		times += 1
		print("time ", times)
		if (turn+1) % 100 == 0:
			times = 0
			#torch.save(pod.mon_reseau.state_dict(), "bestnn.py.pth")
			

	turn +=1

	fenetre.after(10, draw_car)




def draw_checkpoint(check, w):

	global labels

	canvas.delete("check")

	for label in labels:
		label.destroy()

	labels = []

	i = 0
	for c in check:
		rayon = w
		x_centre, y_centre = c.x/DX, c.y/DY
		cercle = canvas.create_oval(x_centre - rayon, y_centre - rayon, x_centre + rayon, y_centre + rayon, fill="orange", tags="check")
			
		carre = canvas.create_rectangle(c.x/DX-25, c.y/DY-25, c.x/DX+25, c.y/DY+25, fill="blue", tags="check")
		label = tk.Label(fenetre, text=str(i))
		label.place(x=c.x/DX, y=c.y/DY) 
		labels.append(label)
		i+=1

turn = 0


moveNN = True
#moveNN = False


# Créer une fenêtre
fenetre = tk.Tk()
fenetre.title("Gamepod")

# Créer un canevas pour dessiner
canvas = tk.Canvas(fenetre, width=dis_width, height=dis_height)
canvas.pack()

labels = []
#draw_checkpoint(checkpoint, 63)


pod = SuperPod() 
vec_env = make_vec_env(SuperPod)

if not moveNN:
	model = PPO("MlpPolicy", vec_env, verbose=1)
	model.learn(total_timesteps=1000000)
	model.save("superpod")
else:
	model = PPO.load("superpod")
	#print(model.policy)

label_error = tk.Label(fenetre, text=str("ERROR = 0.0"))
label_error.place(x=0, y=0) 

state = pod.reset()
next_state =  []
reward = 0
batch_size = 32
nb_reach = 100
done = 0
info = {}
action = 0
prob = 0
value = 0
steps = 0
max_check = 0

next2 = 0
rnext2 = 0
ind_map = 0

times = 0

if moveNN:
	draw_car()
	#Lancer la boucle principale de l'interface graphique
	fenetre.mainloop()
