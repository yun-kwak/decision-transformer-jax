import test_with_cpu  # noqa: F401  # isort:skip

import numpy as np
from absl.testing import absltest
from configs import AtariDefaultOptimalReturn
from gpt import GPT
from torch.utils.data import Dataset
from trainers import AtariTrainer, AtariTrainerConfig

mconf = {
    "vocab_size": 4,
    "n_embd": 128,
    "n_layer": 6,
    "context_len": 3,
    "embd_pdrop": 0.1,
    "transformer_config": {
        "n_embd": 128,
        "attn_config": {
            "n_layer": 6,
            "n_head": 8,
            "n_embd": 128,
            "context_len": 3,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "name": "attn",
        },
        "resid_pdrop": 0.1,
    },
    "max_timestep": 3476,
    "model_type": "reward_conditioned",
    "name": "gpt",
}

tconf = AtariTrainerConfig(
    max_epochs=10,
    batch_size=2,
    learning_rate=0.0006,
    betas=(0.9, 0.95),
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=91674000,
    num_workers=0,
    seed=42,
    model_type="reward_conditioned",
    game="Breakout",
    max_timestep=3476,
)

obs = np.ones((4, 1, 28224), dtype=np.float32)
actions = np.ones((4, 1, 1), dtype=np.int32)
rtgs = np.ones((4, 1, 1), dtype=np.float32)
timesteps = np.ones((4, 1), dtype=np.int32)


class DummyDataset(Dataset):
    def __init__(self, obs, actions, rtgs, timesteps):
        self.obs = obs
        self.actions = actions
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.block_size = 1 * 3

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx], self.rtgs[idx], self.timesteps[idx]


ds = DummyDataset(obs, actions, rtgs, timesteps)


def _fwd(states, actions, rtgs, timestep, is_training):
    model = GPT(**mconf)
    return model(states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=is_training)


class AtariTrainerTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.trainer = AtariTrainer(_fwd, ds, tconf)

    def test_get_returns(self):
        params = self.trainer.init_params()
        self.trainer.get_returns(AtariDefaultOptimalReturn[tconf.game], params, n_epi=1)


if __name__ == "__main__":
    absltest.main()
