import region
from collections import namedtuple

class Exp_Buffer():
    """Buffer initialization
    Parameters
    ----------
    memory : (list)
        Restores the experience namedtuples.
    experience : (namedtuple)
        Restores the experience with name(label).

    Return
    ------
    """
    def __init__(self,batch_size = 4):
        self.memory = []
        self.batch_size = batch_size

        self.experience = namedtuple("Experience", \
                                 field_names = ["state", "contact", "action", "reward", "next_state", "next_contact", "done"])
        self.region_psi_result_set = set()
    def add(self, state, contact, action, reward, next_state, next_contact, done):
        """Add a new experience to memory."""
        e = self.experience(state, contact, action, reward, next_state, next_contact, done)
        self.memory.append(e)


    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        # np.vstack: reshape list to ndarray, column shape (n,1).
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        contacts = torch.from_numpy(np.vstack([e.contact for e in experiences if e is not None])).long().to(device)

        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)\
        action_list = []
        for e in experiences:
            if e is not None:
                action = int(list(e.action.keys())[0])
                action_list.append(action)
        actions = torch.from_numpy(np.vstack(action_list))

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_contacts = torch.from_numpy(np.vstack([e.next_contact for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, contacts, actions, rewards, next_states,next_contacts, dones)
# return a list of namedtuple for experience.
    def get_experience_list(self):
        experience_list = self.memory
        return experience_list