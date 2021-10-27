import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
            self,
            embed_dim,
            src_vocab_size,
            trg_vocab_size,
            src_pad_index,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device
    ):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.src_position_embedding = nn.Embedding(max_len, embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.trg_position_embedding = nn.Embedding(max_len, embed_dim)

        self.transformer = nn.Transformer(
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embed_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_index = src_pad_index
        self.device = device

    def make_src_mask(self, src):
        src_mask = src.transpose(1, 0) == self.src_pad_index
        return src_mask.to(self.device)

    def forward(self, src, trg):
        # src: [src_seq_len, batch]
        # trg: [trg_seq_len, batch]
        src_seq_length, N = src.shape
        trg_seq_length, _ = trg.shape

        # src_position: [src_seq_len, batch]
        src_position = (
            torch.arange(0, src_seq_length)
                .unsqueeze(1)
                .expand(src_seq_length, N)
                .to(self.device)
        )

        # trg_position: [trg_seq_len, batch]
        trg_position = (
            torch.arange(0, trg_seq_length)
                .unsqueeze(1)
                .expand(trg_seq_length, N)
                .to(self.device)
        )

        # embed_src: [src_seq_len, batch, embed_dim]
        embed_src = self.dropout(
            (self.src_embedding(src) + self.src_position_embedding(src_position))
        )

        # embed_trg: [trg_seq_len, batch, embed_dim]
        embed_trg = self.dropout(
            (self.trg_embedding(trg) + self.trg_position_embedding(trg_position))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        # out: [trg_seq_len, batch, embed_dim]
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        # out: [trg_seq_len, batch, embed_dim] -> [trg_seq_len, batch, trg_vocab_size]
        return self.fc_out(out)

# Test the model
#
# transformer = Transformer(EMBED_DIM,
#                           src_vocab_size=10,
#                           trg_vocab_size=10,
#                           src_pad_index=1,
#                           num_heads=NUM_HEADS,
#                           num_encoder_layers=NUM_ENCODER_LAYERS,
#                           num_decoder_layers=NUM_DECODER_LAYERS,
#                           forward_expansion=FORWARD_EXPANSION,
#                           dropout=DROPOUT,
#                           max_len=MAX_LENGTH,
#                           device=DEVICE).to(DEVICE)

# src = torch.LongTensor([
#     [2, 5, 3, 4, 2, 6],
#     [5, 3, 3, 6, 2, 1],
#     [2, 6, 8, 4, 9, 6],
#     [5, 4, 3, 3, 1, 1]
# ]).T.to(DEVICE)
#
# trg = torch.LongTensor([
#     [2, 7, 3, 4, 2, 6, 1],
#     [5, 4, 3, 6, 1, 1, 1],
#     [2, 3, 8, 4, 9, 6, 5],
#     [7, 4, 3, 3, 1, 1, 1]
# ]).T.to(DEVICE)
#
# out = transformer(src, trg)
# print(out, out.shape)
