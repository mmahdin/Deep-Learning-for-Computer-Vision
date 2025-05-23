"""
Implements rnn lstm attention captioning in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import math
import torch.nn as nn
from a4_helper import *
from torch.nn.parameter import Parameter


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from rnn_lstm_attention_captioning.py!')


class FeatureExtractor(object):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, pooling=False, verbose=False,
                 device='cpu', dtype=torch.float32):

        from torchvision import transforms, models
        from torchsummary import summary
        from torchvision.models import MobileNet_V2_Weights

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.device, self.dtype = device, dtype

        # Use the recommended 'weights' parameter
        self.mobilenet = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(device)

        # Remove the last classifier
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        # average pooling
        if pooling:
            # input: N x 1280 x 4 x 4
            self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(4, 4))

        self.mobilenet.eval()
        if verbose:
            summary(self.mobilenet, (3, 112, 112))

    def extract_mobilenet_feature(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape N x 3 x 112 x 112

        Outputs:
        - feat: Image feature, of shape N x 1280 (pooled) or N x 1280 x 4 x 4
        """
        num_img = img.shape[0]

        img_prepro = []
        for i in range(num_img):
            img_prepro.append(self.preprocess(
                img[i].type(self.dtype).div(255.)))
        img_prepro = torch.stack(img_prepro).to(self.device)

        with torch.no_grad():
            feat = []
            process_batch = 500
            for b in range(math.ceil(num_img/process_batch)):
                feat.append(self.mobilenet(img_prepro[b*process_batch:(b+1)*process_batch]
                                           ).squeeze(-1).squeeze(-1))  # forward and squeeze
            feat = torch.cat(feat)

            # add l2 normalization
            F.normalize(feat, p=2, dim=1)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    # Hint: You can use torch.tanh()                                             #
    ##############################################################################
    # Replace "pass" statement with your code
    next_h = torch.tanh(x.mm(Wx) + prev_h.mm(Wh) + b)
    cache = (x, Wx, prev_h, Wh, b, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # Replace "pass" statement with your code
    x, Wx, prev_h, Wh, b, next_h = cache
    dout = dnext_h * (1 - next_h ** 2)  # N*H
    dx = dout.mm(Wx.t())  # N*D
    dWx = x.t().mm(dout)  # D*H
    dprev_h = dout.mm(Wh.t())  # N*H
    dWh = prev_h.t().mm(dout)
    db = dout.sum(dim=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # Replace "pass" statement with your code
    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    prev_h = h0
    cache = []
    for i in range(T):
        next_h, cache_h = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:, i, :] = prev_h
        cache.append(cache_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 

    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # Replace "pass" statement with your code
    N, T, H = dh.shape
    # cache = (x, Wx, prev_h, Wh, b, next_h)
    D = cache[0][0].shape[1]  # shape of dx
    dx = torch.zeros(
        (N, T, D), dtype=cache[0][0].dtype, device=cache[0][0].device)
    dWx = torch.zeros_like(cache[0][1])
    dprev_h = torch.zeros_like(cache[0][2])
    dWh = torch.zeros_like(cache[0][3])
    db = torch.zeros_like(cache[0][4])
    dh0 = torch.zeros((N, H), dtype=dprev_h.dtype, device=dprev_h.device)
    dprev_h_ = torch.zeros_like(dh0)
    for i in range(T-1, -1, -1):
        dx_, dprev_h_, dWx_, dWh_, db_ = rnn_step_backward(
            dh[:, i, :]+dprev_h_, cache[i])
        dx[:, i, :] = dx_
        dWx += dWx_  # sum up because they share
        dWh += dWh_
        db += db_
    dh0 = dprev_h_
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


##############################################################################
# You don't have to implement anything here but it is highly recommended to  #
# through the code as you will write modules on your own later.              #
##############################################################################
class RNN(nn.Module):
    """
    A single-layer vanilla RNN module.

    Arguments for initialization:
    - input_size: Input size, denoted as D before
    - hidden_size: Hidden size, denoted as H before
    """

    def __init__(self, input_size, hidden_size, device='cpu',
                 dtype=torch.float32):
        """
        Initialize a RNN.
        Model parameters to initialize:
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases, of shape (H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = Parameter(torch.randn(input_size, hidden_size,
                                        device=device, dtype=dtype).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn(hidden_size, hidden_size,
                                        device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size,
                           device=device, dtype=dtype))

    def forward(self, x, h0):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Outputs:
        - hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size, embed_size,
                 device='cpu', dtype=torch.float32):
        super().__init__()

        # Register parameters
        self.W_embed = Parameter(torch.randn(vocab_size, embed_size,
                                             device=device, dtype=dtype).div(math.sqrt(vocab_size)))

    def forward(self, x):

        out = None
        ##############################################################################
        # TODO: Implement the forward pass for word embeddings.                      #
        #                                                                            #
        # HINT: This can be done in one line using PyTorch's array indexing.           #
        ##############################################################################
        # Replace "pass" statement with your code
        out = self.W_embed[x]
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V

    Returns a tuple of:
    - loss: Scalar giving loss
    """
    loss = None

    ##############################################################################
    # TODO: Implement the temporal softmax loss function.                        #
    #                                                                            #
    # REQUIREMENT: This part MUST be done in one single line of code!            #
    #                                                                            #
    # HINT: Look up the function torch.functional.cross_entropy, set             #
    # ignore_index to the variable ignore_index (i.e., index of NULL) and        #
    # set reduction to either 'sum' or 'mean' (avoid using 'none' for now).      #
    #                                                                            #
    # We use a cross-entropy loss at each timestep, *summing* the loss over      #
    # all timesteps and *averaging* across the minibatch.                        #
    ##############################################################################
    # Replace "pass" statement with your code
    N, T, V = x.shape
    x_flat = x.reshape(N*T, V)
    y_flat = y.reshape(N*T)
    loss = F.cross_entropy(
        x_flat, y_flat, ignore_index=ignore_index, reduction='sum')
    loss = loss / N
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', device='cpu',
                 ignore_index=None, dtype=torch.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        super().__init__()
        if cell_type not in {'rnn', 'lstm', 'attention'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index

        ##########################################################################
        # TODO: Initialize the image captioning module. Refer to the TODO        #
        # in the captioning_forward function on layers you need to create        #
        #                                                                        #
        # Hint: You may want to check the following pre-defined classes:         #
        # FeatureExtractor, WordEmbedding, RNN, LSTM, AttentionLSTM,             #
        # torch.nn.Linear                                                        #
        #                                                                        #
        # Hint: You can use nn.Linear for both                                   #
        # i) output projection (from RNN hidden state to vocab probability) and  #
        # ii) feature projection (from CNN pooled feature to h0)                 #
        #                                                                        #
        # Hint: In FeatureExtractor, set pooling=True to get the pooled CNN      #
        #       feature and pooling=False to get the CNN activation map.         #
        ##########################################################################
        # Replace "pass" statement with your code
        self.wordvec_dim = wordvec_dim
        self.feat_extract = None
        self.affine = None
        self.rnn = None
        self.affine = nn.Linear(input_dim, hidden_dim).to(
            device=device, dtype=dtype)
        if cell_type == 'rnn' or cell_type == 'lstm':
            self.feat_extract = FeatureExtractor(
                pooling=True, device=device, dtype=dtype)
            # self.affine = nn.Linear(input_dim, hidden_dim).to(device=device, dtype=dtype)
            if cell_type == 'rnn':
                self.rnn = RNN(wordvec_dim, hidden_dim,
                               device=device, dtype=dtype)
            else:
                self.rnn = LSTM(wordvec_dim, hidden_dim,
                                device=device, dtype=dtype)
        elif cell_type == 'attention':
            self.feat_extract = FeatureExtractor(
                pooling=False, device=device, dtype=dtype)
            self.rnn = AttentionLSTM(
                wordvec_dim, hidden_dim, device=device, dtype=dtype)
        else:
            raise ValueError
        nn.init.kaiming_normal_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        self.word_embed = WordEmbedding(
            vocab_size, wordvec_dim, device=device, dtype=dtype)
        self.temporal_affine = nn.Linear(
            hidden_dim, vocab_size).to(device=device, dtype=dtype)
        nn.init.kaiming_normal_(self.temporal_affine.weight)
        nn.init.zeros_(self.temporal_affine.bias)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss. The backward part will be done by torch.autograd.

        Inputs:
        - images: Input images, of shape (N, 3, 112, 112)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Outputs:
        - loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]  # N x T
        captions_out = captions[:, 1:]

        loss = 0.0
        ############################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.                  #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to project the image feature to         #
        #     the initial hidden state $h0$ (for RNN/LSTM, of shape (N, H)) or     #
        #     the projected CNN activation input $A$ (for Attention LSTM,          #
        #     of shape (N, H, 4, 4).                                               #
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL>.                          #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        ############################################################################
        # Replace "pass" statement with your code
        feature = self.feat_extract.extract_mobilenet_feature(
            images)  # N x 1280
        if self.cell_type == 'attention':
            # make it N * 4 * 4 * input_dim
            feature = feature.permute(0, 2, 3, 1)

        h0 = self.affine(feature)  # N x hidden_dim

        if self.cell_type == 'attention':
            h0 = h0.permute(0, 3, 1, 2)  # permute back (N, H, 4, 4)

        x = self.word_embed(captions_in)  # N x T x wordvec_dim
        h = self.rnn(x, h0)  # N x T x H
        score = self.temporal_affine(h)
        loss = temporal_softmax_loss(
            score, captions_out, ignore_index=self._null)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - images: Input images, of shape (N, 3, 112, 112)
        - max_length: Maximum length T of generated captions

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == 'attention':
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine    #
        # transform to the image features. The first word that you feed to         #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to  #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call the `step_forward` from the              #
        # RNN/LSTM/AttentionLSTM module in a loop.                                #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.         #
        # For AttentionLSTM, first project the 1280x4x4 CNN feature activation to  #
        # $A$ of shape Hx4x4. The LSTM initial hidden state and cell state        #
        # would both be A.mean(dim=(2, 3)).                                       #
        ###########################################################################
        # Replace "pass" statement with your code
        feature = self.feat_extract.extract_mobilenet_feature(images)
        A = None
        if self.cell_type == 'attention':
            # make it N * 4 * 4 * input_dim
            feature = feature.permute(0, 2, 3, 1)
        prev_h = self.affine(feature)
        prev_c = torch.zeros_like(prev_h)
        if self.cell_type == 'attention':
            A = prev_h.permute(0, 3, 1, 2)  # permute back
            prev_h = A.mean(dim=(2, 3))
            prev_c = A.mean(dim=(2, 3))

        x = torch.ones((N, self.wordvec_dim), dtype=prev_h.dtype,
                       device=prev_h.device) * self.word_embed(self._start).reshape(1, -1)
        for i in range(max_length):
            next_h = None
            if self.cell_type == 'rnn':
                next_h = self.rnn.step_forward(x, prev_h)
            elif self.cell_type == 'lstm':
                next_h, prev_c = self.rnn.step_forward(x, prev_h, prev_c)
            else:
                attn, attn_weights = dot_product_attention(prev_h, A)
                attn_weights_all[:, i] = attn_weights
                next_h, prev_c = self.rnn.step_forward(x, prev_h, prev_c, attn)

            score = self.temporal_affine(next_h)
            # loss = temporal_softmax_loss(score, captions_out, ignore_index=self._null)
            max_idx = torch.argmax(score, dim=1)
            captions[:, i] = max_idx
            x = self.word_embed(max_idx)
            # print(x.shape)
            prev_h = next_h
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        if self.cell_type == 'attention':
            return captions, attn_weights_all.cpu()
        else:
            return captions


##############################################################################
# LSTM                                                                       #
##############################################################################

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b, attn=None, Wattn=None):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    - attn and Wattn are for Attention LSTM only, indicate the attention input and
      embedding weights for the attention input

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    next_h, next_c = None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use torch.sigmoid() for the sigmoid function.             #
    #############################################################################
    # Replace "pass" statement with your code
    N, H = prev_h.shape
    a = None
    if attn is None:  # regular lstm
        a = x.mm(Wx) + prev_h.mm(Wh) + b
    else:
        a = x.mm(Wx) + prev_h.mm(Wh) + b + attn.mm(Wattn)
    i = torch.sigmoid(a[:, 0:H])
    f = torch.sigmoid(a[:, H:2*H])
    o = torch.sigmoid(a[:, 2*H:3*H])
    g = torch.tanh(a[:, 3*H:4*H])
    next_c = f * prev_c + i * g
    next_h = o * torch.tanh(next_c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    h = None
    # we provide the intial cell state c0 here for you!
    c0 = torch.zeros_like(h0)
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.       #
    #############################################################################
    # Replace "pass" statement with your code
    N, T, D = x.shape
    _, H = h0.shape
    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    prev_h = h0
    prev_c = c0
    for i in range(T):
        next_h, next_c = lstm_step_forward(
            x[:, i, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h[:, i, :] = prev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h


class LSTM(nn.Module):
    """
    This is our single-layer, uni-directional LSTM module.

    Arguments for initialization:
    - input_size: Input size, denoted as D before
    - hidden_size: Hidden size, denoted as H before
    """

    def __init__(self, input_size, hidden_size, device='cpu',
                 dtype=torch.float32):
        """
        Initialize a LSTM.
        Model parameters to initialize:
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = Parameter(torch.randn(input_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn(hidden_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size*4,
                           device=device, dtype=dtype))

    def forward(self, x, h0):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Outputs:
        - hn: The hidden state output
        """
        hn = lstm_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h, prev_c):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        - next_c: The next cell state, of shape (N, H)
        """
        next_h, next_c = lstm_step_forward(
            x, prev_h, prev_c, self.Wx, self.Wh, self.b)
        return next_h, next_c


##############################################################################
# Attention LSTM                                                             #
##############################################################################

def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.
    Inputs:
    - prev_h: The LSTM hidden state from the previous time step, of shape (N, H)
    - A: **Projected** CNN feature activation, of shape (N, H, 4, 4),
         where H is the LSTM hidden state size

    Outputs:
    - attn: Attention embedding output, of shape (N, H)
    - attn_weights: Attention weights, of shape (N, 4, 4)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    #############################################################################
    # TODO: Implement the scaled dot-product attention we described earlier.    #
    # You will use this function for `attention_forward` and `sample_caption`   #
    # HINT: Make sure you reshape attn_weights back to (N, 4, 4)!               #
    #############################################################################
    # Replace "pass" statement with your code
    from math import sqrt
    h_tilt = prev_h.reshape(N, 1, H)
    A_tilt = A.reshape(N, H, -1)
    Matt = (torch.bmm(h_tilt, A_tilt).div(sqrt(H))
            ).reshape(N, -1, 1)  # N * 16 * 1
    Matt_tilt = F.softmax(Matt, dim=1)  # probability
    attn = torch.bmm(A_tilt, Matt_tilt).reshape(N, H)
    attn_weights = Matt_tilt.reshape(N, 4, 4)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return attn, attn_weights


def attention_forward(x, A, Wx, Wh, Wattn, b):
    """
    h0 and c0 are same initialized as the global image feature (meanpooled A)
    For simplicity, we implement scaled dot-product attention, which means in
    Eq. 4 of the paper (https://arxiv.org/pdf/1502.03044.pdf),
    f_{att}(a_i, h_{t−1}) equals to the scaled dot product of a_i and h_{t-1}.

    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data, of shape (N, T, D)
    - A: **Projected** activation map, of shape (N, H, 4, 4)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """

    h = None

    # The initial hidden state h0 and cell state c0 are initialized differently in
    # Attention LSTM from the original LSTM and hence we provided them for you.
    h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
    c0 = h0  # Initial cell state, of shape (N, H)

    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function and dot_product_attention   #
    # function that you just defined.                                           #
    #############################################################################
    # Replace "pass" statement with your code
    N, T, D = x.shape
    _, H = h0.shape
    h = torch.zeros((N, T, H), dtype=h0.dtype, device=h0.device)
    prev_h = h0
    prev_c = c0
    for i in range(T):
        attn, attn_weights = dot_product_attention(prev_h, A)
        next_h, next_c = lstm_step_forward(
            x[:, i, :], prev_h, prev_c, Wx, Wh, b, attn=attn, Wattn=Wattn)
        prev_h = next_h
        prev_c = next_c
        h[:, i, :] = prev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Arguments for initialization:
    - input_size: Input size, denoted as D before
    - hidden_size: Hidden size, denoted as H before
    """

    def __init__(self, input_size, hidden_size, device='cpu',
                 dtype=torch.float32):
        """
        Initialize a LSTM.
        Model parameters to initialize:
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = Parameter(torch.randn(input_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn(hidden_size, hidden_size*4,
                                        device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.Wattn = Parameter(torch.randn(hidden_size, hidden_size*4,
                                           device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size*4,
                           device=device, dtype=dtype))

    def forward(self, x, A):
        """  
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - A: The projected CNN feature activation, of shape (N, H, 4, 4)

        Outputs:
        - hn: The hidden state output
        """
        hn = attention_forward(x, A, self.Wx, self.Wh, self.Wattn, self.b)
        return hn

    def step_forward(self, x, prev_h, prev_c, attn):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)
        - attn: The attention embedding, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        - next_c: The next cell state, of shape (N, H)
        """
        next_h, next_c = lstm_step_forward(x, prev_h, prev_c, self.Wx, self.Wh,
                                           self.b, attn=attn, Wattn=self.Wattn)
        return next_h, next_c
