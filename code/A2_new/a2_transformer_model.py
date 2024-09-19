"""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Author: Raeid Saqur <raeidsaqur@cs.toronto.edu>, Arvid Frydenlund <arvie@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        """
        Compute layer normalization
            y = gamma * (x - mu) / (sigma + eps) + beta where mu and sigma are computed over the feature dimension

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        y = self.gamma * (x - mu) / (sigma + self.eps) + self.beta
        return y


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for both self-attention and cross-attention
    """

    def __init__(
        self,
        num_heads,
        d_model,
        dropout=0.0,
        atten_dropout=0.0,
        store_attention_scores=False,
    ):
        """
        num_heads: int, the number of heads
        d_model: int, the dimension of the model
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        store_attention_scores: bool, whether to store the attention scores for visualization
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        # Assume values and keys are the same size
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # Note for students, for self-attention, it is more efficient to treat q, k, and v as one matrix
        # but this way allows us to use the same attention function for cross-attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(p=atten_dropout)  # applied after softmax

        # applied at the end
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # used for visualization
        self.store_attention_scores = store_attention_scores
        self.attention_scores = None  # set by set_attention_scores

    def set_attention_scores(self, scores):
        """
        A helper function for visualization of attention scores.
        These are stored as attributes so that students do not need to deal with passing them around.

        The attention scores should be given after masking but before the softmax.
        scores: torch.Tensor, shape [batch_size, num_heads, query_seq_len, key_seq_len]
        return: None
        """
        if scores is None:  # for clean up
            self.attention_scores = None
        if self.store_attention_scores and not self.training:
            self.attention_scores = scores.cpu().detach().numpy()

    def attention(self, query, key, value, mask=None):
        """
        Scaled dot product attention
        Hint: the mask is applied before the softmax.
        Hint: attention dropout `self.atten_dropout` is applied to the attention weights after the softmax.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        You are required to call set_attention_scores with the correct tensor before returning from this function.
        The attention scores should be given after masking but before the softmax.

        query: torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        key: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        value: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        mask:  torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True, where masked or None

        return torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        """
        #First we transpose the key tensor to align the dimensions with query tensor for multiplication
        key_T = key.transpose(-2, -1)
        #key_T: torch.Tensor, shape [batch_size, num_heads, d_head, key_seq_len]

        at_s = torch.matmul(query, key_T)
        #at_s: torch.Tensor, [batch_size, num_heads, query_seq_len, key_seq_len]

        at_s = at_s / (self.d_head ** 0.5)
        #No shape change

        #Applying mask at appropriate dimensions
        #Since we are to apply softmax later on, to mask a value in tensor, we should set it to -inf
        #Masking doesnt change the shape
        if mask is not None:
            unsqueeze_mask = mask.unsqueeze(1)
            at_s = at_s.masked_fill(unsqueeze_mask == True, -torch.inf)


        #set attention score
        self.set_attention_scores(at_s)

        #applying softmax (Transforming to weights)
        #No shape change
        at_w = nn.functional.softmax(at_s, dim=-1)
        #at_w: torch.Tensor, [batch_size, num_heads, query_seq_len, key_seq_len]

        #Apply dropout
        #No shape change
        at_w = self.atten_dropout(at_w)

        #Final result
        #at_w: torch.Tensor, [batch_size, num_heads, query_seq_len, key_seq_len]
        #value: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        result = torch.matmul(at_w, value)
        #result: torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]

        return result

    def forward(self, query, key=None, value=None, mask=None):
        """
        If the key and values are None, assume self-attention is being applied.  Otherwise, assume cross-attention.

        Note we only need one mask, which will work for either causal self-attention or cross-attention as long as
        the mask is set up properly beforehand.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        query: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        key: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        value: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        mask: torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True where masked or None

        return: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        """
        #Self attention
        if key is None and value is None:
            key = query
            value = query

        #Linearize
        #No changes in dimension since initialization of linear layer mentions the input and output size which is d_model
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # print('multihead mask: ', mask.shape)

        #Making heads
        #Important note: The size of seq_len doesnt change after reshaping!
        #Size of each tensor, q,k,v is [batch_size, num_heads, query/key_seq_len, d_head]
        v = v.contiguous().view(v.size(dim=0), -1, self.num_heads, self.d_head).transpose(1,2)
        q = q.contiguous().view(q.size(dim=0), -1, self.num_heads, self.d_head).transpose(1,2)
        k = k.contiguous().view(k.size(dim=0), -1, self.num_heads, self.d_head).transpose(1,2)

        #[batch_size, num_heads, query_seq_len, d_head]
        #batch_size = 5
        #num_heads = 4
        #d_model = 8
        #query_seq_len = 3

        #5 * 3 * 8 = 120
        #5, 4, 3, 2

        attention_result = self.attention(q, k, v, mask)
        #attention_result: torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]

        #Reshaping it back to our orignal form
        #do I use self.num_heads * self.d_head vs self.d_model? There is an integer division which might cause an error
        attention_result = attention_result.transpose(1,2)
        attention_result = attention_result.contiguous().view(q.size(0), -1, self.num_heads * self.d_head)
        #attention_result: torch.Tensor, shape [batch_size, query_seq_len, d_model]

        #Linearize
        #No change in shape
        attention_result = self.out_linear(attention_result)

        #Dropout
        attention_result = self.dropout(attention_result)
        #No change in shape

        return attention_result


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.f = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Compute the feedforward sublayer.
        Dropout is applied after the activation function and after the second linear layer

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        h = self.w_1(x)
        h = self.f(h)
        h = self.dropout1(h)
        h = self.w_2(h)
        h = self.dropout2(h)

        return h


class TransformerEncoderLayer(nn.Module):
    """

    Idea if we can give this init done, then the students can fill in the decoder init in the same way but add in cross attention


    Performs multi-head self attention and FFN with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
            Hint:  be careful about zeroing out tokens.  How does this affect the softmax?
        """
        super(TransformerEncoderLayer, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads

        self.ln1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )

        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

    def pre_layer_norm_forward(self, x, mask):
        """
        x: torch.Tensor, the input to the layer
        mask: torch.Tensor, the mask to apply to the attention
        Hint:  should only require two or three lines of code
        """
        layernorm1 = self.ln1(x)
        multihead_attention = self.self_attn(layernorm1, layernorm1, layernorm1, mask)
        multihead_attention = multihead_attention + x
        layernorm2 = self.ln2(multihead_attention)
        feedforward = self.ff(layernorm2)
        feedforward = feedforward + multihead_attention
        return feedforward

    def post_layer_norm_forward(self, x, mask):
        #Do I use the same ln1 and ln2 again in post_layer_norm_forward?
        multihead_attention = self.self_attn(x, x, x, mask)
        multihead_attention = multihead_attention + x
        layernorm1 = self.ln1(multihead_attention)
        feedforward = self.ff(layernorm1)
        feedforward = feedforward + layernorm1
        layernorm2 = self.ln2(feedforward)
        return layernorm2

    def forward(self, x, mask):
        if self.is_pre_layer_norm:
            return self.pre_layer_norm_forward(x, mask)
        else:
            return self.post_layer_norm_forward(x, mask)


class TransformerEncoder(nn.Module):
    """
    Stacks num_layers of TransformerEncoderLayer and applies layer norm at the correct place.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        super(TransformerEncoder, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout, is_pre_layer_norm
                )
            )
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        """
        x: torch.Tensor, the input to the encoder
        mask: torch.Tensor, the mask to apply to the attention
        """
        if not self.is_pre_layer_norm:
            x = self.norm(x)
        for layer in self.layers:
            x = layer(x, mask)
        if self.is_pre_layer_norm:
            x = self.norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Performs multi-head self attention, multi-head cross attention, and FFN,
    with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        is_pre_layer_norm: bool, whether to apply layer norm before or after each sublayer

        Please use the following attribute names 'self_attn', 'cross_attn', and 'ff' and any others you think you need.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads

        #Multihead attention
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout)
        self.cross_attn = MultiHeadAttention(num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout)

        #Feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

        #LayerNorm
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def pre_layer_norm_forward(self, x, mask, src_x, src_mask):
        layernorm1 = self.ln1(x)
        self_attention = self.self_attn(layernorm1, layernorm1, layernorm1, mask)
        self_attention = self_attention + x
        layernorm2 = self.ln2(self_attention)
        #Use src_x in cross_attentton
        cross_attention = self.cross_attn(layernorm2, src_x, src_x, src_mask)
        cross_attention = cross_attention + self_attention
        layernorm3 = self.ln3(cross_attention)
        feedforward = self.ff(layernorm3)
        feedforward = feedforward + cross_attention
        return feedforward

    def post_layer_norm_forward(self, x, mask, src_x, src_mask):
        self_attention = self.self_attn(x, x, x, mask)
        self_attention = self_attention + x
        layernorm1 = self.ln1(self_attention)
        cross_attention = self.cross_attn(layernorm1, src_x, src_x, src_mask)
        cross_attention = cross_attention + layernorm1
        layernorm2 = self.ln2(cross_attention)
        feedforward = self.ff(layernorm2)
        feedforward = feedforward + layernorm2
        layernorm3 = self.ln3(feedforward)
        return layernorm3

    def forward(self, x, mask, src_x, src_mask):
        if self.is_pre_layer_norm:
            return self.pre_layer_norm_forward(x, mask, src_x, src_mask)
        else:
            return self.post_layer_norm_forward(x, mask, src_x, src_mask)

    def store_attention_scores(self, should_store=True):
        self.self_attn.store_attention_scores = should_store
        self.cross_attn.store_attention_scores = should_store

    def get_attention_scores(self):
        return self.self_attn.attention_scores, self.cross_attn.attention_scores


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        super(TransformerDecoder, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout, is_pre_layer_norm
                )
            )
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)  # logit projection

    def forward(self, x, mask, src_x, src_mask, normalize_logits: bool = False):
        """
        x: torch.Tensor, the input to the decoder
        mask: torch.Tensor, the mask to apply to the attention
        src_x: torch.Tensor, the output of the encoder
        src_mask: torch.Tensor, the mask to apply to the attention
        normalize_logits: bool, whether to apply log_softmax to the logits

        Returns the logits or log probabilities if normalize_logits is True

        Hint: look at the encoder for how pre/post layer norm is handled
        """
        if not self.is_pre_layer_norm:
            x = self.norm(x)
        for layer in self.layers:
            x = layer(x, mask, src_x, src_mask)
        if self.is_pre_layer_norm:
            x = self.norm(x)

        projection_logit = self.proj(x)

        if normalize_logits:
            log_probs = nn.functional.log_softmax(projection_logit, dim=-1)
            return log_probs
        else:
            return projection_logit

        
    def store_attention_scores(self, should_store=True):
        for layer in self.layers:
            layer.store_attention_scores(should_store)

    def get_attention_scores(self):
        """
        Return the attention scores (self-attention, cross-attention) from all layers
        """
        scores = []
        for layer in self.layers:
            scores.append(layer.get_attention_scores())
        return scores


class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TransformerEmbeddings, self).__init__()
        self.lookup = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        x: torch.Tensor, shape [batch_size, seq_len] of int64 in range [0, vocab_size)
        return torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.lookup(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].requires_grad_(False)
        x = x + pos  # Add the position encoding to original vector x
        return self.dropout(x)


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        padding_idx: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
        no_src_pos: bool = False,
        no_tgt_pos: bool = False,
    ):
        super(TransformerEncoderDecoder, self).__init__()
        """
        src_vocab_size: int, the size of the source vocabulary
        tgt_vocab_size: int, the size of the target vocabulary
        padding_idx: int, the index of the pad token
        num_layers: int, the number of layers
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        is_pre_layer_norm: bool, whether to apply layer norm before or after each sublayer
        no_src_pos: bool, whether to skip positional encoding for the source
        no_tgt_pos: bool, whether to skip positional encoding for the target
        """

        self.src_embed = TransformerEmbeddings(src_vocab_size, d_model)
        if no_src_pos:
            self.src_pe = None
            print("Warning: no positional encoding for the source")
        else:
            self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_embed = TransformerEmbeddings(tgt_vocab_size, d_model)
        if no_tgt_pos:
            self.tgt_pe = None
            print("Warning: no positional encoding for the target")
        else:
            self.tgt_pe = PositionalEncoding(d_model, dropout)

        self.encoder = TransformerEncoder(
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout,
            atten_dropout,
            is_pre_layer_norm,
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout,
            atten_dropout,
            is_pre_layer_norm,
        )

        self.padding_idx = padding_idx

    def create_pad_mask(self, tokens):
        """
        Create a padding mask using pad_idx (an attribute of the class)
        Hint: respect the output shape

        tokens: torch.Tensor, [batch_size, seq_len]
        return: torch.Tensor, [batch_size, 1, seq_len] where True means to mask, and on the same device as tokens
        """
        #check if tokens are padded or not
        pad_mask = tokens.eq(self.padding_idx)
        #Add one dimesion
        pad_mask = pad_mask.unsqueeze(1).to(tokens.device)

        return pad_mask

    @staticmethod
    def create_causal_mask(tokens):
        """
        Create a causal (upper) triangular mask
        Hint: respect the output shape and this can be done via torch.triu

        tokens: torch.Tensor, [batch_size, seq_len]
        pad_idx: int, the index of the pad token
        return: torch.Tensor, [1, seq_len, seq_len] where True means to mask, and on the same device as tokens
            Hint, if seq_len = 5, then the mask should look like:
                tensor([[[False,  True,  True,  True,  True],
                         [False, False,  True,  True,  True],
                         [False, False, False,  True,  True],
                         [False, False, False, False,  True],
                         [False, False, False, False, False]]])
            and make sure to set the correct dtype and device
        """
        seq_len = tokens.size(1)
        temp_tensor = torch.ones(seq_len, seq_len).to(tokens.device)
        #Only consider elemnts above the diagnol
        causal_mask = torch.triu(temp_tensor, diagonal=1)
        #Make it True False
        causal_mask = causal_mask.bool()
        #add dimension
        causal_mask = causal_mask.unsqueeze(0)

        return causal_mask

    def get_src_embeddings(self, src):
        """
        Get the non-contextualized source embeddings
        src: torch.Tensor, [batch_size, src_seq_len]
        return: torch.Tensor, [batch_size, src_seq_len, d_model]
        """
        src_x = self.src_embed(src)
        if self.src_pe is not None:
            src_x = self.src_pe(src_x)
        return src_x

    def get_tgt_embeddings(self, tgt):
        """
        Get the non-contextualized target embeddings
        tgt: torch.Tensor, [batch_size, tgt_seq_len]
        return: torch.Tensor, [batch_size, tgt_seq_len, d_model]
        """
        tgt_x = self.tgt_embed(tgt)
        if self.tgt_pe is not None:
            tgt_x = self.tgt_pe(tgt_x)
        return tgt_x

    def forward(self, src, tgt, normalize_logits: bool = False):
        """
        src: torch.Tensor, [batch_size, src_seq_len]
        tgt: torch.Tensor, [batch_size, tgt_seq_len]
        normalize_logits: bool, whether to apply log_softmax to the logits
        return: torch.Tensor, [batch_size, tgt_seq_len, tgt_vocab_size] of logits or log probabilities
        """
        #print(tgt.shape)
        src_mask = self.create_pad_mask(src)
        #print(src_mask.shape)
        tgt_mask = self.create_pad_mask(tgt)
        tgt_causal_mask = self.create_causal_mask(tgt)

        emb_src = self.get_src_embeddings(src)
        emb_tgt = self.get_tgt_embeddings(tgt)
        encoded = self.encoder(emb_src, src_mask)
        decoded = self.decoder(emb_tgt, tgt_causal_mask, encoded, src_mask, normalize_logits)

        return decoded


    # decoding methods
    @staticmethod
    def all_finished(current_generation, eos_token, max_len=100):
        """
        Check if all the current generation is finished
        current_generation: torch.Tensor, [batch_size, seq_len]
        eos_token: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: bool, True if all the current generation is finished
        """
        return (
            torch.all(torch.any(current_generation == eos_token, dim=-1), dim=0).item()
            or current_generation.shape[-1] >= max_len
        )

    @staticmethod
    def initialize_generation_sequence(src, target_sos):
        """
        Initialize the generation by returning the initial input for the decoder.
        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        return: torch.Tensor, [batch_size, 1] on the same device as src and of time int64 filled with target_sos
        """
        return (
            torch.zeros(src.shape[0], 1).fill_(target_sos).type_as(src).to(src.device)
        )

    @staticmethod
    def concatenate_generation_sequence(tgt_generation, next_token):
        """
        Concatenate the next token to the current generation
        tgt_generation: torch.Tensor, [batch_size, seq_len]
        next_token: torch.Tensor, [batch_size, 1]
        return: torch.Tensor, [batch_size, seq_len + 1]
        """
        #print(tgt_generation.shape)
        #print(next_token.shape)

        answer = torch.cat([tgt_generation, next_token], dim=-1)

        # print('concatenate_generation_sequence')
        # print(tgt_generation.shape)
        # print(answer.shape)

        return answer

    def pad_generation_sequence(self, tgt_generation, target_eos):
        """
        Replace the generation past the end of sequence token with the end of sequence token.

        This is useful for:
            1) finalizing the generation for greedy decoding
            2) helping with intermediate steps in beam search

        tgt_generation: torch.Tensor, [batch_size, seq_len] or [batch_size, k * k, seq_len]
        target_eos: int, the end of sequence token
        return: torch.Tensor, [batch_size, seq_len] or [batch_size, k * k, seq_len]
        """
        lengths = (
            tgt_generation == target_eos
        )  # [batch_size, seq_len] or [batch_size, k * k, seq_len]
        # deal with case where eos was not found  [batch_size, seq_len + 1] or [batch_size, k * k, seq_len + 1]

        lengths = torch.cat(
            [lengths, torch.ones(*lengths.shape[:-1], 1).bool().to(lengths.device)],
            dim=-1,
        )
        # find the first eos token
        a = (
            torch.arange(lengths.shape[-1], device=tgt_generation.device)
            .unsqueeze(0)
            .int()
        )  # [1, seq_len + 1]
        if len(lengths.shape) == 3:
            a = a.unsqueeze(1)
        lengths = (
            torch.where(lengths, lengths * a, torch.ones_like(lengths) * torch.inf)
        ).min(dim=-1)[0]
        # replace tokens past the eos token with the padding token
        mask = a[..., :-1] > lengths.unsqueeze(
            -1
        )  # [batch_size, seq_len] or [batch_size, k * k, seq_len]
        return tgt_generation.masked_fill(mask, self.padding_idx)

    def greedy_decode(self, src, target_sos, target_eos, max_len=100):
        """
        Do not call the encoder more than once, or you will lose marks.
        The model calls must be batched and the only loop should be over the sequence length, or you will lose marks.
        It will also make evaluating the debugging the model difficult if you do not follow these instructions.

        Hint: use torch.argmax to get the most likely token at each step and
        concatenate_generation_sequence to add it to the sequence.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, seq_len]
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
            Hint: use the pad_generation_sequence function
        """
        #Encode (Called once)
        emb_src = self.get_src_embeddings(src)
        src_mask = self.create_pad_mask(src)
        src_encoded = self.encoder(emb_src,src_mask)

        #Initialize target generation
        tgt_generation = self.initialize_generation_sequence(src, target_sos)

        #Loop over sequence length
        for i in range(1, max_len):

            tgt_emb = self.get_tgt_embeddings(tgt_generation)
            tgt_mask = self.create_causal_mask(tgt_generation)
            
            #(log) prob
            #No mask for tgt
            #Batched? Its in a loop, but when I call decoder each time, it will still be a batched model call
            log_prob = self.decoder(tgt_emb, tgt_mask, src_encoded, src_mask, normalize_logits=True)
            #print(log_prob.shape)
            #Get the last token prob
            #Maintaining batchsize and dmodel
            log_prob = log_prob[:, -1, :]
            #print('log prob dim: ', log_prob.shape)

            #last token in sequence select
            prediction =torch.argmax(log_prob, dim=-1).unsqueeze(1)
            #print(prediction.shape)
            #print('prediction :', prediction.shape)

            #concat
            tgt_generation = self.concatenate_generation_sequence(tgt_generation, prediction)
            #print(tgt_generation.shape)


            #Optimization
            if self.all_finished(tgt_generation, target_eos, max_len):
                break

        #Hint
        tgt_generation = self.pad_generation_sequence(tgt_generation, target_eos)

        return tgt_generation

    @staticmethod
    def expand_encoder_for_beam_search(src_x, src_mask, k):
        """
        Beamsearch will process `batches` of size `batch_size * k` so we need to expand the encoder outputs
        so that we can process the beams in parallel.

        Expand the encoder outputs for beam search to be of size [batch_size * k, ...]
        src_x: torch.Tensor, [batch_size, src_seq_len, d_model]
        src_mask: torch.Tensor, [batch_size, 1, src_seq_len]
        k: int, the beam size
        return: torch.Tensor, [batch_size * k, src_seq_len, d_model], [batch_size * k, 1, src_seq_len]
        """
        #repeat interleaves was used instead of repeat
        new_src_x = src_x.repeat_interleave(k, dim=0)
        new_mask = src_mask.repeat_interleave(k, dim=0)
    
        return new_src_x, new_mask

    @staticmethod
    def repeat_and_reshape_for_beam_search(t, k, expan, batch_size):
        """
        Repeat the tensor for beam search expan times to be of size [batch_size * k, expan, cur_len]
        and then reshape to [batch_size, k * expan, cur_len]

        t: torch.Tensor, [batch_size * k, cur_len]
        k: int, the beam size
        expan: int, the expansion size
        batch_size: int, the batch size
        return: torch.Tensor, [batch_size, k * expan, cur_len]
        """
        #1) Repeat the tensor for beam search expan times to be of size [batch_size * k, expan, cur_len]
        #print('expan: ', expan)
        #print('input shape: ', t.shape)
        new_t = t.unsqueeze(1).repeat(1, expan, 1)
        #print('new_t: ', new_t.shape)
        
        #2) reshape to [batch_size, k * expan, cur_len]
        new_t = new_t.view(batch_size, k * expan, -1)
        #print(new_t.shape)
        #cur_len is maintained

        return new_t

    def initialize_beams_for_beam_search(
        self, src, target_sos, target_eos, max_len=100, k=5
    ):
        """
        This function will initialize the beam search by taking the first decoder step and using the top-k outputs
        to initialize the beams.

        Here we want to end up with a tensor of shape [batch_size * k, 2] for the input token sequence
        and a tensor of shape [batch_size * k, 2] of log probabilities of the sequences.
        2 is the sequence dimension and is 2 because of the sos token and the first real token.

        This involves the following steps:

            1) Initializes the input sequence with the start of sequence token  with shape [batch_size, 1]
            2) Takes the first step of the decoder and the get the log probabilities, [batch_size, 1, vocab_size]
            3) Please ensure that the end of sequence token is not predicted in the first step.
                Hint: set the log probabilities of the end of sequence token to -inf
            4) Gets the top-k predictions, [batch_size, k]
            5) Initializes the log probabilities of the sequences, [batch_size * k, 1]
            6) Creates the beam tensors with the top-k predictions and log probabilities, [batch_size  * k, 2]
               (i.e., two tensors with this shape).
            7) Expands the encoder outputs for beam search to be of size [batch_size * k, ...]
                Hint: use the expand_encoder_for_beam_search function

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size * k, 2], the token sequences
                torch.Tensor, [batch_size * k, 2], the log probabilities of the sequences
                torch.Tensor, [batch_size * k, src_seq_len, d_model], the expanded encoder outputs
                torch.Tensor, [batch_size * k, 1, src_seq_len], the expanded encoder mask
        """

        device = src.device
        
        #1) Initializes the input sequence with the start of sequence token  with shape [batch_size, 1]
        tgt_generation = self.initialize_generation_sequence(src, target_sos)

        emb_src = self.get_src_embeddings(src)
        emb_tgt = self.get_tgt_embeddings(tgt_generation)

        #2) Takes the first step of the decoder and the get the log probabilities, [batch_size, 1, vocab_size]
        src_mask = self.create_pad_mask(src)
        encoded_src = self.encoder(emb_src, src_mask)

        tgt_mask = self.create_causal_mask(tgt_generation)
        
        #Only dealing with sos so I dont use mask
        log_probs = self.decoder(emb_tgt, tgt_mask, encoded_src, src_mask, normalize_logits=True)

        #3) Please ensure that the end of sequence token is not predicted in the first step.
        #Assuming target_eos is the index for target_eos in d_model
        log_probs[:, :, target_eos] = -torch.inf

        #4) Gets the top-k predictions, [batch_size, k]
        #I think the shape is wrong here. which I corrected using squeeze
        top_probs, top_indices = torch.topk(log_probs, k, dim=-1)
        top_probs = top_probs.squeeze(1)
        top_indices = top_indices.squeeze(1)
        
        #5) Initializes the log probabilities of the sequences, [batch_size * k, 1]
        reshaped_probs = top_probs.view(-1, 1).to(device)
        reshaped_indices = top_indices.view(-1, 1).to(device)
        #print(reshaped_indices.dtype)

        #6) Creates the beam tensors with the top-k predictions and log probabilities, [batch_size  * k, 2]
        init_log_probs = torch.zeros((src.size(0) * k, 1)).to(device)
        init_target_sos = (torch.ones((src.size(0) * k, 1)) * target_sos).to(device)   

        beam_probs = torch.cat([init_log_probs, reshaped_probs], dim=1)
        beam_idx = torch.cat([init_target_sos, reshaped_indices], dim=1)
        beam_idx = beam_idx.to(torch.int64)

        #7) Expands the encoder outputs for beam search to be of size [batch_size * k, ...]
                #Hint: use the expand_encoder_for_beam_search function

        expanded_src_x, expanded_src_mask = self.expand_encoder_for_beam_search(encoded_src, src_mask, k)

        return beam_idx, beam_probs, expanded_src_x, expanded_src_mask

    def pad_and_score_sequence_for_beam_search(
        self, tgt_generation, seq_log_probs, target_eos
    ):
        """
        This function will pad the sequences with eos and seq_log_probs with corresponding zeros.
        It will then get the score of each sequence by summing the log probabilities.

        Note assume that we want this to work for generic shapes of the input where the last dimension is the sequence.

        Hint: use pad_generation_sequence and self.padding_idx.

        tgt_generation: torch.Tensor, [..., seq_len]
        seq_log_probs: torch.Tensor, [..., seq_len]
        target_eos: int, the end of sequence token
        return: torch.Tensor, [..., seq_len], the padded token sequences
                torch.Tensor, [...., seq_len], the log probabilities of the sequences
                torch.Tensor, [...], the summed scores of the sequences
        """
        #self.padding_idx

        #print('tgt_generation: ', tgt_generation.shape)
        padded_seq = self.pad_generation_sequence(tgt_generation, target_eos)
        #print('padded_seq: ', padded_seq.shape)
        padded_prob = self.pad_generation_sequence(seq_log_probs, self.padding_idx)
        #print('padded_prob: ', padded_prob.shape)
        padded_prob[padded_seq == self.padding_idx] = 0
        # eos_mask = padded.eq(target_eos).unsqueeze(-1)
        # mask = torch.cumsum(eos_mask, dim=-1).eq(0)

        # seq_probs = seq_log_probs.masked_fill(~mask, 0.0)
        scores = padded_prob.sum(dim=-1)

        return padded_seq, padded_prob, scores

    def finalize_beams_for_beam_search(self, top_beams, device):
        """
        This function will take a list of top beams of length batch_size, where each element is a tensor of some length
        and return a padded tensor of the top beams.  Use self.padding_idx for the padding.

        top_beams: list of torch.Tensor, each of shape [seq_len_i]
        device: torch.device, the device to put the tensor on
        return: torch.Tensor, [batch_size, max_seq_len]
        """
        # max_seq_len
        max_seq_len = max(len(beam) for beam in top_beams)

        padded_top_beams = torch.full((len(top_beams), max_seq_len), self.padding_idx, dtype=torch.long, device=device).to(device)

        for idx, beam in enumerate(top_beams):
            padded_top_beams[idx, :len(beam)] = beam

        return padded_top_beams

    def beam_search_decode(self, src, target_sos, target_eos, max_len=100, k=5):
        """
        This processes the batch in parallel.  Finished beams are removed from the batch and not processed further.

        This needs to keep track of a list of finished candidates along with currently alive beams to prevent
        the search from stopping prematurely with duplicate beams.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, max_seq_len] of the mostly likely beam
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
        """

        # get the initial first predictions and initialize the beams with them
        batch_size = src.shape[0]  # original batch size
        (
            tgt_generation,
            seq_log_probs,
            src_x,
            src_mask,
        ) = self.initialize_beams_for_beam_search(
            src, target_sos, target_eos, max_len, k
        )

        finished_sequences = [[] for _ in range(batch_size)]
        finished_scores = [
            [] for _ in range(batch_size)
        ]  # averaged log probs i.e score
        is_finished = [False] * batch_size

        # use to keep track of which sequence in batch are still being processed
        cur_sentence_ids = torch.arange(batch_size, device=src.device)  # [batch_size]
        cur_batch_size = batch_size

        # expansion size for each beam, 2 so that we can possibly get k finished sequences and k alive
        # this is used in place of vocab_size for efficiency
        expan = 2 * k
        while not all(is_finished) and tgt_generation.shape[-1] < max_len:
            tgt_mask = self.create_causal_mask(tgt_generation)
            tgt_x = self.get_tgt_embeddings(tgt_generation)

            log_probs = self.decoder(
                tgt_x, tgt_mask, src_x, src_mask, normalize_logits=True
            )
            log_probs, pred = torch.topk(
                log_probs[:, -1, :], expan, dim=1
            )  # [batch_size * k, expan]

            # reshape to cur to [batch_size, k * expan, cur_len] and new to [batch_size, k * expan, 1]
            tgt_generation = self.repeat_and_reshape_for_beam_search(
                tgt_generation, k, expan, cur_batch_size
            )
            seq_log_probs = self.repeat_and_reshape_for_beam_search(
                seq_log_probs, k, expan, cur_batch_size
            )
            log_probs = log_probs.reshape(cur_batch_size, k * expan, 1)
            pred = pred.reshape(cur_batch_size, k * expan, 1)

            # expansion to [batch_size, k * expan, cur_len + 1]
            tgt_generation = self.concatenate_generation_sequence(tgt_generation, pred)
            seq_log_probs = torch.cat(
                [seq_log_probs, log_probs], dim=-1
            )  # or concatenate_generation_sequence

            #  masking to length and score by summing
            (
                tgt_generation,
                seq_log_probs,
                scores,
            ) = self.pad_and_score_sequence_for_beam_search(
                tgt_generation, seq_log_probs, target_eos
            )

            # sort and get top k of the current expan candidates
            scores, topk = torch.topk(scores, expan, dim=1)  # [batch_size, k * expan]

            # gather the top expan candidates sorted by score, [batch_size, expan, cur_len + 1]
            tgt_generation = torch.gather(
                tgt_generation,
                1,
                topk.unsqueeze(-1).expand(-1, -1, tgt_generation.shape[-1]),
            )
            seq_log_probs = torch.gather(
                seq_log_probs,
                1,
                topk.unsqueeze(-1).expand(-1, -1, seq_log_probs.shape[-1]),
            )

            # split into finished and alive
            has_eos = (tgt_generation == target_eos).any(dim=-1)  # [batch_size, expan]
            alive_idxs = []
            has_not_finished_idxs = []  # otherwise remove from batch for efficiency
            for b in range(cur_batch_size):
                original_b = cur_sentence_ids[b].item()  # original sentence id
                alive_idxs_b = []
                for beam in range(expan):
                    if has_eos[b, beam]:
                        finished_sequences[original_b].append(
                            tgt_generation[b, beam, ...].cpu()
                        )
                        finished_scores[original_b].append(scores[b, beam].item())
                    else:
                        alive_idxs_b.append(beam)

                if len(finished_sequences[original_b]) > 1:  # sort finished sequences
                    z = list(
                        zip(finished_scores[original_b], finished_sequences[original_b])
                    )
                    z = sorted(z, key=lambda x: x[0], reverse=True)
                    finished_scores[original_b] = [
                        x[0] for x in z[:k]
                    ]  # cut to k if more than k finished
                    finished_sequences[original_b] = [x[1] for x in z[:k]]

                # these conditions only work assuming an monotonic increase in score (length normalization breaks it)
                has_finished = False
                if len(
                    finished_sequences[original_b]
                ):  # if the most probable finished is more probable than top alive
                    best_finished = max(finished_scores[original_b])
                    if best_finished > scores[b, alive_idxs_b[0]]:
                        has_finished = True
                elif (
                    len(finished_sequences[original_b]) == k
                ):  # if all finished more probable than top alive
                    # This block will never execute, since it is a subset of the if condition assuming that k != 0.
                    # Arvie: This is redundant but should not cause incorrect output. This alternative stopping
                    # condition was left in 1) potentially we could have returned k finished sequences, as is more
                    # traditional for a beamsearch function, 2) I was trying alternative scoring methods which
                    # required this stopping condition.  Apologies if it made the code more confusing.
                    worst_finished = min(finished_scores[original_b])
                    if worst_finished > scores[b, alive_idxs_b[0]]:
                        has_finished = True
                if not has_finished:
                    has_not_finished_idxs.append(b)
                    alive_idxs.append(alive_idxs_b[:k])

            if len(has_not_finished_idxs) == 0:  # all finished
                break

            # gather sequences that still need decoding, this is done for efficiency
            has_not_finished_idxs = torch.tensor(
                has_not_finished_idxs, dtype=torch.long, device=src.device
            )
            tgt_generation = torch.index_select(
                tgt_generation, 0, has_not_finished_idxs
            )
            seq_log_probs = torch.index_select(seq_log_probs, 0, has_not_finished_idxs)
            src_shape = src_x.shape
            src_x = src_x.reshape(cur_batch_size, k, src_shape[-2], src_shape[-1])
            src_mask = src_mask.reshape(cur_batch_size, k, -1)
            src_x = torch.index_select(src_x, 0, has_not_finished_idxs)
            src_mask = torch.index_select(src_mask, 0, has_not_finished_idxs)
            cur_batch_size = len(has_not_finished_idxs)
            src_x = src_x.reshape(cur_batch_size * k, src_shape[-2], src_shape[-1])
            src_mask = src_mask.reshape(cur_batch_size * k, 1, src_shape[-2])
            cur_sentence_ids = torch.index_select(
                cur_sentence_ids, 0, has_not_finished_idxs
            )

            # gather the top k alive beams
            alive_idxs = torch.tensor(
                alive_idxs, dtype=torch.long, device=tgt_generation.device
            )

            tgt_generation = torch.gather(
                tgt_generation,
                1,
                alive_idxs.unsqueeze(-1).expand(-1, -1, tgt_generation.shape[-1]),
            )
            seq_log_probs = torch.gather(
                seq_log_probs,
                1,
                alive_idxs.unsqueeze(-1).expand(-1, -1, seq_log_probs.shape[-1]),
            )

            # reshape to [batch_size * k, cur_len]
            tgt_generation = tgt_generation.reshape(cur_batch_size * k, -1)
            seq_log_probs = seq_log_probs.reshape(cur_batch_size * k, -1)

        if not all(is_finished):  # take top alive if no finished
            tgt_generation = tgt_generation.reshape(cur_batch_size, k, -1)
            for b in range(cur_batch_size):
                original_b = cur_sentence_ids[b].item()
                if len(finished_sequences[original_b]) == 0:  # just take top alive
                    g = tgt_generation[b, 0, ...]
                    finished_sequences[original_b].append(g.cpu())

        return self.finalize_beams_for_beam_search(
            [x[0] for x in finished_sequences], src_x.device
        )

    def beam_search_decode_slow(self, src, target_sos, target_eos, max_len=100, k=5):
        """
        Slow version which does not batch the beam search. This is useful for confirming debugging and understanding.

        Consider the summed log probability of the sequence (including eos) when scoring and sorting the candidates.

        This needs to keep track of a list of finished candidates along with currently alive beams to prevent
        the search from stopping prematurely with duplicate beams.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, max_seq_len] of the mostly likely beam
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
        """

        # print('Warning: this is a slow version of beam search and should only be run for debugging.')
        # get the initial first predictions and initialize the beams with them
        batch_size = src.shape[0]  # original batch size

        (
            tgt_generation,
            seq_log_probs,
            src_x,
            src_mask,
        ) = self.initialize_beams_for_beam_search(
            src, target_sos, target_eos, max_len, k
        )

        # reshape so we can loop over the batch dimension
        tgt_generation = tgt_generation.reshape(batch_size, k, -1)
        seq_log_probs = seq_log_probs.reshape(batch_size, k, -1)
        src_x = src_x.reshape(batch_size, k, src_x.shape[-2], src_x.shape[-1])
        src_mask = src_mask.reshape(
            batch_size, k, 1, -1
        )  # note the extra dim for the mask

        top_beams = []
        expan = 2 * k  # expansion size, 2 * k for k finished and k alive,
        # this is used in place of vocab_size for efficiency
        for b in range(batch_size):
            # slice out the current sequence from the batch
            tgt_generation_b = tgt_generation[b, ...]
            seq_log_probs_b = seq_log_probs[b, ...]
            src_x_b = src_x[b, ...]
            src_mask_b = src_mask[b, ...]

            finished_sequences_b = []
            finished_scores_b = []

            for i in range(1, max_len):
                tgt_mask = self.create_causal_mask(tgt_generation_b)
                tgt_x = self.get_tgt_embeddings(tgt_generation_b)
                log_probs = self.decoder(
                    tgt_x, tgt_mask, src_x_b, src_mask_b, normalize_logits=True
                )
                log_probs, pred = torch.topk(log_probs[:, -1, :], expan, dim=1)

                # reshape to cur to [k * expan, cur_len] and new to [k * expan, 1]
                tgt_generation_b = self.repeat_and_reshape_for_beam_search(
                    tgt_generation_b, k, expan, 1
                ).squeeze(0)
                seq_log_probs_b = self.repeat_and_reshape_for_beam_search(
                    seq_log_probs_b, k, expan, 1
                ).squeeze(0)
                log_probs = log_probs.reshape(k * expan, 1)
                pred = pred.reshape(k * expan, 1)

                # expansion to [batch_size, k * expan, cur_len + 1]
                tgt_generation_b = self.concatenate_generation_sequence(
                    tgt_generation_b, pred
                )
                seq_log_probs_b = torch.cat([seq_log_probs_b, log_probs], dim=-1)

                # masking to length and score by summing
                (
                    tgt_generation_b,
                    seq_log_probs_b,
                    scores,
                ) = self.pad_and_score_sequence_for_beam_search(
                    tgt_generation_b, seq_log_probs_b, target_eos
                )

                # sort and get top k of the current expan candidates
                scores, topk = torch.topk(scores, expan, dim=0)  # [k * expan]

                # gather the top expan candidates sorted by score, [expan, cur_len + 1]
                tgt_generation_b = torch.index_select(tgt_generation_b, 0, topk)
                seq_log_probs_b = torch.index_select(seq_log_probs_b, 0, topk)

                # split into finished and alive and determine if we are finished
                has_eos = (tgt_generation_b == target_eos).any(dim=-1)  # [expan]
                alive_idxs_b = []
                for beam in range(expan):
                    if has_eos[beam]:
                        finished_sequences_b.append(tgt_generation_b[beam, ...].cpu())
                        finished_scores_b.append(scores[beam].item())
                    else:
                        alive_idxs_b.append(beam)
                if len(finished_sequences_b) > 1:  # sort finished sequences
                    z = list(zip(finished_scores_b, finished_sequences_b))
                    z = sorted(z, key=lambda x: x[0], reverse=True)
                    finished_scores_b = [
                        x[0] for x in z[:k]
                    ]  # cut to k if more than k finished
                    finished_sequences_b = [x[1] for x in z[:k]]

                # these conditions only work assuming an monotonic increase in score (length normalization breaks it)
                if len(
                    finished_sequences_b
                ):  # finished if most probable finished is more probable than top alive
                    best_finished = max(finished_scores_b)
                    if best_finished > scores[alive_idxs_b[0]]:
                        break
                elif (
                    len(finished_sequences_b) == k
                ):  # finished if all finished more probable than top alive
                    worst_finished = min(finished_scores_b)
                    if worst_finished > scores[alive_idxs_b[0]]:
                        break

                # gather the top k alive beams
                alive_idxs_b = torch.tensor(
                    alive_idxs_b[:k], dtype=torch.long, device=tgt_generation.device
                )
                tgt_generation_b = torch.index_select(tgt_generation_b, 0, alive_idxs_b)
                seq_log_probs_b = torch.index_select(seq_log_probs_b, 0, alive_idxs_b)

            if len(finished_sequences_b) == 0:  # take top alive if no finished
                finished_sequences_b.append(tgt_generation_b[0, ...].cpu())
            top_beams.append(finished_sequences_b[0])

        return self.finalize_beams_for_beam_search(top_beams, src_x.device)

    def store_attention_scores(self, should_store=True):
        self.decoder.store_attention_scores(should_store)

    def get_attention_scores(self):
        return self.decoder.get_attention_scores()
