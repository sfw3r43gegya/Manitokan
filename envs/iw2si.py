from .minigrid import *
# from gym_minigrid.register import register
COLORS = ["purple", "yellow", "orange", 'cyan']
class iw2si(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self,
            size=8,
            episode_limit=0,
            seed=1337,
            agent_view_size=0,
             reward_sparse=None,
             reward_local=None,
            only_sparse = False,
            only_immediate = False,
            credit_easy_af = False,
            reward_coop=False,
            completion_signal = False,
            key_signal=False,
                 window=2,
             n_agents=2,
             n_keys = 1,
            p =0.5,
                 name=''
                 ):
        super().__init__(
            grid_size=size,
            episode_limit=episode_limit,
            seed=seed,
            agent_view_size=3,
            reward_sparse=reward_sparse,
            reward_local=reward_local,
            only_sparse=only_sparse,
            only_immediate=only_immediate,
            reward_coop=reward_coop,
            credit_easy_af=credit_easy_af,
            completion_signal=completion_signal,
            key_signal=key_signal,
            p=p,
            n_agents=n_agents,
            n_keys=n_keys
            ,
            name=name

        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width)
       # self.grid.vert_wall(splitIdx, 0)

        # Place a goal in the bottom-right corner


        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        for agent_idx in range(self.n_agents):
            # Randomize the starting agent position and direction
            self.place_agent(agent_idx, size=(splitIdx, height))

        # Place a door in the wall
        self.prev_doors = []






        for door in range(self.n_agents):

            self.place_obj(
                obj=Door(COLORS[door],
                              is_locked=True,
                              mano=door,),
                top=(0, 0),
                size= (width, height))

        # Place a yellow key on the left side
        for key in range(self.n_keys):
            self.place_obj(
                obj=Key('blue'),
                top=(0, 0),
                size= (width, height))

        self.splitIdx = splitIdx

        self.mission = "credit assignment"


    def _placeGoal(self):
        splitIdx = self.splitIdx
        room_idx = self._rand_int(0, 2)

        if not room_idx:
            self.place_obj(Goal(), top=(0, 0),
                           size=(splitIdx, self.height))
        else:
            self.place_obj(Goal(), top=(splitIdx + 1, 0),
                           size=(self.width - splitIdx - 1, self.height))




