import torch
from torch import nn

from deep_likeliness.tokenizer import encode, get_vocab_size


class RNNEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, bidirectional=False):
        """
        Torch Module for encoding SMILES to latent space
        :param hidden_size: GRU hidden size
        :param num_layers: Number of stacked GRU Cells
        :param latent_size: dimension of latent space
        :param bidirectional: GRU bi-directionality
        """
        super(RNNEncoder, self).__init__()

        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, 1)
        )

    def encode(self, sm_list):
        """
        Maps smiles onto a latent space.
        :param sm_list: tuple fo strings of length batch representing SMILES
        :return: (Cuda) FloatTensor of shape (Batch, 2 * latent space) repsenting RNN encoded SMILES in latent space
        RNN
        """

        # tokens: LongTensor of shape (batch, latent)
        # lens: Integer List of length batch
        # self.embs: (latent, batch) => (latent, batch, hidden_size)
        # self.rnn: FloatTensor (latent, batch, hidden_size) => FloatTensors (latent, batch, hidden),((2*)num_layer, batch, hidden)
        # outputs: FloatTensor of shape (batch, hidden)
        # self.final_mlp: FloatTensor (batch, hidden) => FloatTensor (batch, 2*latent_size)

        tokens, lens = encode(sm_list)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        outputs = self.rnn(self.embs(to_feed))[0]
        outputs = outputs[
            lens, torch.arange(len(lens))
        ]  # select final hidden before sequence padding for each element in batch

        mean_std = self.final_mlp(outputs)

        return mean_std, outputs

    def score_list(self, smiles):
        with torch.no_grad():
            scores, obs = self.encode(smiles)
            scores = scores.squeeze().numpy()
            obs = obs.squeeze().numpy()

        return scores, obs
