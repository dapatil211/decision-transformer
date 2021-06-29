from decision_transformer.models.model import TrajectoryModel
import torch


class GRUModel(TrajectoryModel):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length,
        test_max_length=-1,
        max_ep_len=4096,
        action_tanh=True,
        use_time_encoding=False,
        use_attention=False,
        shuffle=False,
        shuffle_last=False,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        if test_max_length == -1:
            self.test_max_length = max_length
        else:
            self.test_max_length = test_max_length
        self.hidden_size = hidden_size
        self.use_time_encoding = use_time_encoding
        self.use_attention = use_attention
        self.shuffle = shuffle
        self.shuffle_last = shuffle_last

        self.gru = torch.nn.GRU(hidden_size, hidden_size, **kwargs)

        if self.use_time_encoding:
            self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = torch.nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = torch.nn.Sequential(
            *(
                [torch.nn.Linear(hidden_size, self.act_dim)]
                + ([torch.nn.Tanh()] if action_tanh else [])
            )
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        tlens=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if tlens is None:
            torch.full((batch_size,), seq_length)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        if self.shuffle:
            # indices = torch.randperm()
            range_tensor = (
                torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
            )
            mask = (
                range_tensor >= tlens.unsqueeze(1)
                if self.shuffle_last
                else range_tensor > tlens.unsqueeze(1)
            )
            rand_values = torch.rand(batch_size, seq_length)
            rand_values[mask] = 1
            indices = torch.argsort(rand_values)
            state_embeddings = state_embeddings[indices]
            action_embeddings = action_embeddings[indices]
            returns_embeddings = returns_embeddings[indices]

        if self.use_time_encoding:
            time_embeddings = self.embed_timestep(timesteps)
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )

        stacked_inputs = self.embed_ln(stacked_inputs)

        gru_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            stacked_inputs, 3 * tlens, batch_first=True, enforce_sorted=False
        )
        x = self.gru(gru_inputs)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        return_preds = self.predict_return(
            x[:, 2]
        )  # predict next return given state and action
        state_preds = self.predict_state(
            x[:, 2]
        )  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # these will come as tensors on the correct device
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.test_max_length is not None:
            states = states[:, -self.test_max_length :]
            actions = actions[:, -self.test_max_length :]
            returns_to_go = returns_to_go[:, -self.test_max_length :]
            timesteps = timesteps[:, -self.test_max_length :]
            tlens = torch.tensor([states.shape[1]])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        (
                            states.shape[0],
                            self.test_max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.test_max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.test_max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.zeros(
                        (timesteps.shape[0], self.test_max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            tlens = None
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, tlens=tlens, **kwargs
        )
        return action_preds[0, -1]


# GRU
#   get_batch - 1 hr
#   rest of function - 15 mins
#   testing - 1hr
# shuffling
#   get batch - 30 mins
# vary context length
#   get_batch - 15 mins
# test time only use context = 1
#   add test_max_length - 15 mins
