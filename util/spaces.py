import numpy as np
import melee
import code

class ObservationSpace:
    # maybe best to just load a yml file for certain settings?
    def __init__(self, stage):
        self.current_frame = 0
        self.done = False
        self.previous_observation = None
        self.current_observation = None
        self.stage = stage
        self.total_reward = 0

    def _reset(self):
        self.current_frame = 0
        self.done = False
        self.previous_observation = None
        self.current_observation = None

    def get_stocks(self, gamestate):
        stocks = [gamestate.players[i].stock for i in list(gamestate.players.keys())]
        return np.array([stocks]).T  # players x 1

    def get_damage(self, gamestate):
        damage = [gamestate.players[i].percent for i in list(gamestate.players.keys())]
        return np.array([damage]).T  # players x 1

    def get_actions(self, gamestate):
        actions = [gamestate.players[i].action.value for i in list(gamestate.players.keys())]
        action_frames = [gamestate.players[i].action_frame for i in list(gamestate.players.keys())]
        return np.array([actions, action_frames]).T  # players x 3

    def get_hitstun(self, gamestate):
        hitstun_frames_left = [gamestate.players[i].hitstun_frames_left for i in list(gamestate.players.keys())]
        return np.array([hitstun_frames_left]).T

    def get_positions(self, gamestate):
        x_positions = [gamestate.players[i].x for i in list(gamestate.players.keys())]
        y_positions = [gamestate.players[i].y for i in list(gamestate.players.keys())]
        return np.array([x_positions, y_positions]).T # players x 2

    def _convert_to_cell_coords(self, positions, cell_size=5):
        # Probably the easiest solution to reduce observable space. Could also
        #   have some sort of dynamic cell system that affords greater accuracy
        #   in certain places. Not sure how to implement. 

        # can do the whole thing at once
        return np.floor(positions/cell_size)

    def __call__(self, gamestate):
        """ pull out relevant info from gamestate """
        self.current_frame +=1 
        info = None
        reward = 0

        positions = self._convert_to_cell_coords(self.get_positions(gamestate))
        actions = self.get_actions(gamestate)
        hitstun = self.get_hitstun(gamestate)
        damage = self.get_damage(gamestate)
        stocks = self.get_stocks(gamestate)

        self.current_observation = np.concatenate((positions, actions, hitstun, damage, stocks), axis=1).astype(int)
        
        # this is fancy one liner that just says if the player(s) with the 
        #   fewest stocks sum to zero, the game is over. Doesn't cover teams.
        self.done = not self.current_observation[np.argsort(self.current_observation[:, -1])][::-1][1:, -1]

        if self.current_frame > 85 and not self.done:
            # reward/penalize based on delta damage
            reward = self.current_observation[:, -2] - self.previous_observation[:, -2]
            reward = np.flip(reward)
            reward = np.clip(reward, 0, None)

            self.total_reward += reward

        if self.current_observation is not None:
            self.previous_observation = self.current_observation

        return self.current_observation, reward, self.done, info