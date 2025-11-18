/// RNN Encoder-Decoder Implementation in Rust
/// Example implementation using bidirectional LSTM cells

pub struct RNNEncoder {
    rnn: RNN,  // Base RNN cell (e.g., bidirectional LSTM)
}

impl RNNEncoder {
    pub fn new(input_size: usize, output_size: usize, hidden_size: usize, num_layers: usize, forget_bias: &mut f64) -> Self {
        let lstm = LSTM::new(hidden_size, hidden_size / 2, forget_bias);
        RNNEncoder {
            rnn: bidir_lstm(num_layers, lstm, lstm),
        }
    }

    pub fn encode(&mut self, input_seq: &Vec<i32>) -> Vec<f64> {
        // Convert each input symbol to a sequence of one-hot encoded tokens
        let symbol_tokens = input_seq.iter().map(|symbol| {
            let mut tokens = Vec::new();
            for token in ngram::tokenize(symbol.to_string(), 2) {
                match embeddings.get(&token.to_owned()) {
                    Some(&embedding) => {
                        for embedding_element in embedding {
                            tokens.push(if embedding_element == token { 1.0 } else { 0.0 });
                        }
                    }
                    None => {}
                };
            }
            tokens
        }).collect::<Vec<_>>();

        let mut encoder_state = rnn.zeros((symbol_tokens.len(), rnn.cells)); // Initialize RNN state and cell variables
        let mut output = vec![];

        for (i, symbols) in symbol_tokens.iter().zip(0..).enumerate() {
            // Feed the one-hot encoded symbol sequence through the encoder RNN
            rnn.forward(&mut encoder_state[i], &symbols);
            // Flatten and concatenate hidden states along the time axis to obtain a vector representation of the input sequence
            let flattened_hidden_states = rnn.flatten(encoder_state[i]);
            let context = tf::tanh(tf::reduce_sum(tf::reshape(&flattened_hidden_states, [None, encoder.cells]), 1)); // Apply a tanh nonlinearity
            output.push(context);
        }

        output
    }
}

pub struct RNNDecoder {
    rnn: RNN, // Base RNN cell (e.g., bidirectional LSTM)
}

impl RNNDecoder {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, num_layers: usize, forget_bias: &mut f64) -> Self {
        let lstm = LSTM::new(hidden_size, hidden_size / 2, forget_bias);
        RNNDecoder {
            rnn: bidir_lstm(num_layers, lstm, lstm),
        }
    }

    pub fn decode(&mut self, src: &Vec<i32>, tgt: &Vec<i32>) -> Vec<i32> {
        // Compute attention weights matrix and encode source sequence using encoder from above
        let src_sequence_length = src.len();
        let target_indices = vec![src_sequence_length as i32 - 1]; // Initialize target indices array with start of sequence index
        let encoded_src_sequence = self.encode(&src);

        let mut cell_state = rnn.zeros((encoded_src_sequence.len(), rnn.cells));   // Initialize RNN state and cell variables
        let mut decoded_symbols = vec![];

        for t in 0..tgt.len() {
            // Compute log-probs for each vocabulary element at this timestep
            let log_probs = tf::gather(&self.compute_logits(&encoded_src_sequence, &mut target_indices), t as i32) + &self.rnn.biases;
            // Sample from the probability distribution to get the predicted symbol for this timestep
            let index = mathutils::argmax(log_probs);

            let pred_symbol = embeddings[index as usize];

            if t == src.len() - 1 {
                continue;
            }

            decoded_symbols.push(pred_symbol);
        }

        decoded_symbols
    }
}
