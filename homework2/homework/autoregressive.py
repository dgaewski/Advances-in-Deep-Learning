import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()

        self.n_tokens = n_tokens
        self.d_latent = d_latent

        #use positonal embedding with the tokens
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.pos_embedding = torch.nn.Embedding(600, d_latent)  # 600 = 20*30 max
        self.start_token = torch.nn.Parameter(torch.zeros(1, 1, d_latent))

        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_latent,
                nhead=4,
                dim_feedforward=d_latent * 4,
                dropout=0.0,        
                batch_first=True,   # (B, seq, d_latent)
            ),
            num_layers=4,           # 4 layers for the transformer
        )

        self.output_proj = torch.nn.Linear(d_latent, n_tokens)

        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        B, h, w = x.shape
        seq_len = h * w

        #flatten the tensor to a long sequence
        x = x.reshape(B, seq_len)

        # embed tokens
        x = self.token_embedding(x)

        # generate position indices [0, 1, 2, ..., seq_len-1] at runtime
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)   # (1, seq)

        #add the position infomation
        x = x + self.pos_embedding(positions)

        #create a mask for the tokens
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        #define the start token
        start = self.start_token.expand(B, 1, self.d_latent)  # (B, 1, d_latent)

        #this is where the shift happens - we add the start token and drop off the last token
        x = torch.cat([start, x[:, :-1, :]], dim=1)           # (B, seq, d_latent)

        #pass through the transformer with the casual mask applied
        x = self.transformer(x, mask=mask, is_causal=True)

        #we project to logits over all tokens
        x = self.output_proj(x)

        #rehape the output back to image size
        output = x.reshape(B, h, w, self.n_tokens)

        return output, {}
        # raise NotImplementedError()


    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        
        """
        Autoregressively generate B images of shape (B, h, w) token by token.

        At each step we:
          1. Run forward on the tokens generated so far (padded to full length)
          2. Take the predicted distribution at the current position
          3. Sample one token
          4. Store it and move to the next position
        """
        seq_len = h * w

        # initialize tokens with all zeros
        tokens = torch.zeros(B, seq_len, dtype=torch.long, device=device)

        for i in range(seq_len):
            
            x_in = tokens.reshape(B, h, w)

            with torch.no_grad():
                logits, _ = self.forward(x_in)  # (B, h, w, n_tokens)

            # flatten spatial dims to index position i
            logits_i = logits.reshape(B, seq_len, self.n_tokens)[:, i, :]  # (B, n_tokens)

            # Sample from the distribution (multinomial sampling)
            probs = torch.softmax(logits_i, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

            # Store the sampled token at position i
            tokens[:, i] = next_token

        return tokens.reshape(B, h, w)

        #raise NotImplementedError()
