from melee_env.agents.basic import Agent, AgentChooseCharacter, is_defeated
from melee import enums


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class A2C(Agent):
	def __init__(self):
		super().__init__()
		self.character = enums.Character.FOX

	@is_defeated
	def act(self, observation, action_space):
		self.action = action_space.sample()
