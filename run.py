from melee import enums
from melee_env.env import MeleeEnv
from melee_env.agents.util import ActionSpace
from melee_env.agents.basic import NOOP, Human
from util.a2cfox import A2C
from util.spaces import ObservationSpace

players = [Human(), NOOP(enums.Character.FOX)]
stage = enums.Stage.FINAL_DESTINATION

observation_space = ObservationSpace(stage)

env = MeleeEnv(
    "/home/david/Games/melee/roms/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso",
    players,
    ActionSpace(),
    observation_space,
    fast_forward=False, 
    blocking_input=True)

env.start()
episodes = 3; reward = 0; done = False

for episode in range(episodes):
    observation, reward, done, info = env.setup(stage)
    while not done:
        print(observation)
        for i in range(len(players)):
            players[i].act(observation, env.action_space)

        observation, reward, done, info = env.step()
