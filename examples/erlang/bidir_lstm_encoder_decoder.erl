%%% Bidirectional LSTM Cell (base RNN architecture)
%%% Example implementation of RNN Encoder-Decoder with Attention Mechanism
-module(bidir_lstm_encoder_decoder).
-export([encode/1, decode/2]).

%%% Bidirectional LSTM Cell (base RNN architecture)
bidir_lstm:new(InputSize, OutputSize, ForgetBias, CellSize) ->
    % Create two forward and backward LSTM cells of the same size as the input sequence
    ForwardLSTM = lstm:new(InputSize, OutputSize/2, ForgetBias, CellSize),
    BackwardLSTM = lstm:new(InputSize, OutputSize/2, ForgetBias, CellSize),

    % Connect the forward and backward cells together to create a bidirectional cell
    BidirLSTM = fun([h0_, c0_]) ->
        for i <- 1, size(h0_) of
            # Pass the current hidden state through both the forward and backward cells
            [hn_, cn_] = lstm:forward(ForwardLSTM, h0_[i]),
            [hn2_, cn2_] = lstm:forward(BackwardLSTM, c0_[length(c0_) - i + 1]),

            % Compute the final hidden state and cell state by combining the forward and backward outputs
            hn_ = nalgebra:horzcat([hn_, hn2_]),
            cn_ = nalgebra:horzcat([cn_, cn2_]),
        end,
        [hn_, cn_];
    end.

    BidirLSTM.

%%% Encoder RNN
encode(InputSymbols) ->
    % Initialize encoder cell state variables
    Encoder = bidir_lstm:new(1, EmbeddingDim),
    EncodedVecs = lists:new(),
    InputSizes = lists:new().

    for symbol <- InputSymbols of
        symbol_size = ngram:tokenize(symbol, MaxNgram),  % Convert symbol to sequence of input tokens
        [EmbeddingVector] = embeddings:lookup(symbol_size),   % Look up the input token embedding vectors

        % Initialize cell state variables for each direction of the LSTM
        [h0_, c0_] = lstm:zeros([length(embedding_vector), Encoder#cells]),
        InputSizes = append(InputSizes, 1),

        for i <- 1, length(embedding_vector) of
            % Feed each embedding vector element through the encoder cell
            [hn_, cn_] = lstm:forward(Encoder, h0_, embedding_vector[i]),

            % Add the resulting hidden state and cell state to the running totals
            h0_ = nalgebra:horzcat([h0_, hn_]),
            c0_ = nalgebra:horzcat([c0_, cn_]),
        end,

        EncodedVec = reshape2d(hn_, 1, EmbeddingDim),   % Flatten the final hidden state to obtain the encoded vector representation of the input sequence
        EncodedVecs = append(EncodedVecs, EncodedVec),
    od.

    EncodedVecs.

%%% Decoder RNN
decode([SrcSymbol], [TgtSymbol]) ->
    % Initialize decoder cell state variables and attention weights matrix
    AttentionWeightsMatrix = nalgebra:eye(max_encoder_sequence_length),   % Initialize the attention weights matrix as the identity matrix
    Decoder = bidir_lstm:new(EmbeddingDim, VocabSize),
    DecodedSymbols = lists:new(),
    CellStates = lists:new().

    % Encode the source sequence using the encoder RNN from above
    EncodedSrcSequence = encode([SrcSymbol]);

    % Loop over each timestep in the target sequence
    for t <- 1, length(TgtSymbol) of
        % Initialize hidden state and cell state variables for the decoder LSTM
        [h0_, c0_] = lstm:zeros([EncodedSrcSequence#size, Decoder#cells]),

        % Look up the embedding vector for the target symbol at this timestep
        [EmbeddingVector] = embeddings:lookup(TgtSymbol[t]),

        % Calculate the attention weights for this timestep
        AttentionWeightsMatrix = calculate_attention_weights(AttentionWeightsMatrix, EncodedSrcSequence),

        for i <- 1, length(AttentionWeightsMatrix) of
            % Compute the context vector by taking a weighted average of all encoded vectors, applying a tanh nonlinearity, and flattening the result to a vector
            ContextVector = math:tanh(nalgebra:sum(apply2fun(mul, AttentionWeightsMatrix[i], EncodedSrcSequence), 1)),

            % Perform a forward pass through the decoder LSTM with the current context vector as input
            [hn_, cn_] = lstm:forward(Decoder, h0_, ContextVector),

            % Add the resulting hidden state and cell state to the running totals
            h0_ = nalgebra:horzcat([h0_, hn_]),
            c0_ = nalgebra:horzcat([c0_, cn_]),
        end,

        % Sample from the probability distribution over the vocabulary to get the predicted symbol at this timestep
        LogProbs = nalgebra:dot(Decoder#weights * Decoder#biases - h0_ * Decoder#forget_bias, 1),
        ProbDist = nalgebra:softmax(LogProbs),

        % Choose the index of the maximum probability and look up the corresponding symbol
        PredictedIndex = math:argmax(probdist),
        [PredictedSymbol] = embeddings:lookup([predicted_index]),

        % Add the predicted symbol to the output list
        DecodedSymbols = append(DecodedSymbols, PredictedSymbol),
    od.

    # Return the sequence of predicted symbols
    DecodedSymbols.
