import torch
from torch import Tensor, nn


class LSTMLanguageModel(nn.Module):
    """
    A language model based on LSTM architecture.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the word embeddings.
        hidden_dim (int): Dimension of the LSTM hidden state.
        embedding (nn.Embedding): Embedding layer.
        lstm (nn.LSTM): LSTM layer.
        fc (nn.Linear): Fully connected layer for output logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pretrained_embeddings: Tensor | None = None,
        freeze: bool = False,
    ) -> None:
        """
        Initializes the LSTMLanguageModel.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            hidden_dim (int): Dimension of the LSTM hidden state.
            pretrained_embeddings (Optional[Tensor]): Pretrained embeddings tensor.
            freeze (bool): Whether to freeze the pretrained embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        if pretrained_embeddings is not None:
            self.vocab_size, self.embedding_dim = pretrained_embeddings.shape
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (Tensor): Input tensor containing token IDs.

        Returns:
            Tensor: Logits for each token in the sequence.
        """
        # Embedding lookup
        embeddings = self.embedding(input_ids)

        # LSTM forward pass
        lstm_out, _ = self.lstm(embeddings)

        # Fully connected layer for logits
        logits = self.fc(lstm_out)

        return logits
