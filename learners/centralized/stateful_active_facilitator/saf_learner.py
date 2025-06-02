from learners.centralized.stateful_active_facilitator.saf import SAF
from learners.centralized.stateful_active_facilitator.saf_buffer import ReplayBufferImageObs

class SAFLearner:
    def __init__(self,  args, ):
        self.args = args

        self.name = "saf"

        self.policy = SAF(observation_space= args.obs_shape,
                          action_space=args.n_actions,
                          state_space=args.state_shape,
                          params=self.args)

        self.buffer = ReplayBufferImageObs(observation_space = args.obs_shape,
                                           action_space =args.n_actions,
                                           params = args,
                                           device = self.args.device)




    def get_actions(self, x, state, action_mask=None, actions=None, x_old=None):

        action, logprob, _, value, _ = self.policy.get_action_and_value(x,
                                                                        state,
                                                                        action_mask,
                                                                        None,
                                                                        x_old)

        return action, logprob, value


    def train(self, advantages, returns):

       return self.policy.train_step( self.buffer, advantages, returns)

