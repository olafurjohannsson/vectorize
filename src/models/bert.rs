use crate::models::ModelConfig;
use crate::models::components::{ActivationFunction, Embedding, LayerNorm, Linear};
use crate::tensor::Tensor;
use crate::weights::ModelWeights;

pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
    config: ModelConfig,
}

struct BertEncoder {
    layers: Vec<BertLayer>,
}

struct BertPooler {
    dense: Linear,
}

struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: f32,
}

struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

struct BertIntermediate {
    dense: Linear,
    intermediate_act_fn: ActivationFunction,
}

struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertModel {
    pub fn new(weights: ModelWeights) -> Self {
        let config = weights.config.clone();

        // initialize embeddings
        let embeddings = BertEmbeddings::new(&weights, &config);

        // initialize encoder layers
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(&weights, &config, layer_idx));
        }
        let encoder = BertEncoder { layers };

        Self {
            embeddings,
            encoder,
            pooler: None,
            config,
        }
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Tensor {
        let embedding_output = self.embeddings.forward(input_ids, None);

        // Prepare attention mask for all layers
        // Convert from [batch, seq_len] with 1s and 0s
        // to extended attention mask for adding to attention scores
        let extended_attention_mask = self.get_extended_attention_mask(attention_mask);

        // Pass through encoder
        let encoder_output = self
            .encoder
            .forward(embedding_output, Some(&extended_attention_mask));

        // Mean pooling for sentence embeddings
        self.mean_pooling(encoder_output, attention_mask)
    }

    fn get_extended_attention_mask(&self, attention_mask: &Tensor) -> Tensor {
        attention_mask.clone()
    }

    fn mean_pooling(&self, hidden_states: Tensor, attention_mask: &Tensor) -> Tensor {
        // hidden_states: [batch_size, seq_len, hidden_size]
        // attention_mask: [batch_size, seq_len]
        println!("hidden_states shape: {:?}", hidden_states.shape());
        println!("attention_mask shape: {:?}", attention_mask.shape());
        // We need to expand attention_mask from [batch, seq] to [batch, seq, hidden]
        //  add a dimension: [batch, seq] -> [batch, seq, 1]
        let mask_expanded = attention_mask.unsqueeze(-1);
        println!("mask_expanded after unsqueeze: {:?}", mask_expanded.shape());
        // broadcast [batch, seq, 1] -> [batch, seq, hidden_size]
        let hidden_size = hidden_states.shape()[2];
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];

        let mask_expanded = mask_expanded.broadcast_to(&[batch_size, seq_len, hidden_size]);

        // mask and sum over sequence dimension
        let masked = hidden_states.mul(&mask_expanded);
        let sum_embeddings = masked.sum(&[1], false); // Sum over seq_len dimension

        // denominator (sum of mask) for averaging
        let sum_mask = mask_expanded.sum(&[1], false).clamp_min(1e-9);

        // Divide to get mean
        sum_embeddings.div(&sum_mask)
    }
}

impl crate::models::EmbeddingModel for BertModel {
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Tensor {
        self.forward(input_ids, attention_mask)
    }

    fn get_config(&self) -> &ModelConfig {
        &self.config
    }
}

impl BertEncoder {
    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask);
        }
        hidden_states
    }
}

impl BertEmbeddings {
    fn new(weights: &ModelWeights, config: &ModelConfig) -> Self {
        Self {
            word_embeddings: Embedding::new(
                weights
                    .get_tensor("embeddings.word_embeddings.weight")
                    .clone(),
            ),
            position_embeddings: Embedding::new(
                weights
                    .get_tensor("embeddings.position_embeddings.weight")
                    .clone(),
            ),
            token_type_embeddings: Embedding::new(
                weights
                    .get_tensor("embeddings.token_type_embeddings.weight")
                    .clone(),
            ),
            layer_norm: LayerNorm::new(
                weights.get_tensor("embeddings.LayerNorm.weight").clone(),
                weights.get_tensor("embeddings.LayerNorm.bias").clone(),
                config.layer_norm_eps,
            ),
            dropout: 0.1,
        }
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Tensor {
        // input_ids: [batch_size, seq_length]
        let batch_size = input_ids.shape()[0];
        let seq_length = input_ids.shape()[1];

        // Get word embeddings: [batch_size, seq_length, hidden_size]
        let inputs_embeds = self.word_embeddings.forward(input_ids);

        // Create position IDs [seq_length] and broadcast for batch
        let position_ids = Tensor::arange(0, seq_length as i64);
        // Expand to [batch_size, seq_length] if needed
        let position_ids = if batch_size > 1 {
            position_ids
                .unsqueeze(0)
                .broadcast_to(&[batch_size, seq_length])
        } else {
            position_ids.unsqueeze(0)
        };
        let position_embeddings = self.position_embeddings.forward(&position_ids);

        // Token type embeddings (0 for single sentence)
        let zeros_tensor;
        let token_type_ids = if let Some(ids) = token_type_ids {
            ids
        } else {
            zeros_tensor = Tensor::zeros(vec![batch_size, seq_length]);
            &zeros_tensor
        };
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);

        // Add all embeddings together
        let embeddings = inputs_embeds
            .add(&position_embeddings)
            .add(&token_type_embeddings);

        // Apply layer norm
        self.layer_norm.forward(&embeddings)
    }
}

impl BertLayer {
    fn new(weights: &ModelWeights, config: &ModelConfig, layer_idx: usize) -> Self {
        Self {
            attention: BertAttention {
                self_attention: BertSelfAttention::new(weights, config, layer_idx),
                output: BertSelfOutput::new(weights, config, layer_idx),
            },
            intermediate: BertIntermediate::new(weights, config, layer_idx),
            output: BertOutput::new(weights, config, layer_idx),
        }
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        // Self attention
        let attention_output = self
            .attention
            .self_attention
            .forward(hidden_states, attention_mask);
        let attention_output = self
            .attention
            .output
            .forward(&attention_output, hidden_states);

        // Feed forward
        let intermediate_output = self.intermediate.forward(&attention_output);
        let layer_output = self.output.forward(&intermediate_output, &attention_output);

        layer_output
    }
}

impl BertSelfAttention {
    fn new(weights: &ModelWeights, config: &ModelConfig, layer_idx: usize) -> Self {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = hidden_size / num_attention_heads;

        let prefix = format!("encoder.layer.{}.attention.self", layer_idx);

        Self {
            query: Linear::new(
                weights
                    .get_tensor(&format!("{}.query.weight", prefix))
                    .clone(),
                weights
                    .get_tensor(&format!("{}.query.bias", prefix))
                    .clone(),
            ),
            key: Linear::new(
                weights
                    .get_tensor(&format!("{}.key.weight", prefix))
                    .clone(),
                weights.get_tensor(&format!("{}.key.bias", prefix)).clone(),
            ),
            value: Linear::new(
                weights
                    .get_tensor(&format!("{}.value.weight", prefix))
                    .clone(),
                weights
                    .get_tensor(&format!("{}.value.bias", prefix))
                    .clone(),
            ),
            num_attention_heads,
            attention_head_size,
        }
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        // Linear projections
        let query_layer = self.transpose_for_scores(&self.query.forward(hidden_states));
        let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states));
        let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states));

        // query_layer, key_layer, value_layer: [batch, num_heads, seq_len, head_size]

        // Attention scores: [batch, num_heads, seq_len, seq_len]
        let attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2));
        let attention_scores =
            attention_scores.div_scalar((self.attention_head_size as f32).sqrt());

        let attention_scores = if let Some(mask) = attention_mask {
            // mask is [batch, seq_len]
            //  reshape it to [batch, 1, 1, seq_len] for broadcasting
            //broadcast to [batch, num_heads, seq_len, seq_len]

            // First, expand mask from [batch, seq_len] to [batch, 1, 1, seq_len]
            let mask_expanded = mask
                .unsqueeze(1) // [batch, 1, seq_len]
                .unsqueeze(2); // [batch, 1, 1, seq_len]

            // Create attention mask values: 0 for valid, -10000 for invalid
            // Assuming mask has 1 for valid positions and 0 for padding
            let mask_value = mask_expanded
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .mul_scalar(-10000.0);

            // Add mask to attention scores
            attention_scores.add(&mask_value)
        } else {
            attention_scores
        };

        // Normalize with softmax
        let attention_probs = attention_scores.softmax(-1);

        // Apply attention to values
        let context_layer = attention_probs.matmul(&value_layer);

        // Reshape back
        self.transpose_from_scores(&context_layer)
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Tensor {
        let mut new_shape = x.shape().to_vec();
        new_shape.pop();
        new_shape.push(self.num_attention_heads);
        new_shape.push(self.attention_head_size);

        x.reshape(&new_shape).transpose(-2, -3)
    }

    fn transpose_from_scores(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[2];

        x.transpose(-2, -3).reshape(&[
            batch_size,
            seq_len,
            self.num_attention_heads * self.attention_head_size,
        ])
    }
}

impl BertSelfOutput {
    fn new(weights: &ModelWeights, config: &ModelConfig, layer_idx: usize) -> Self {
        let prefix = format!("encoder.layer.{}.attention.output", layer_idx);

        Self {
            dense: Linear::new(
                weights
                    .get_tensor(&format!("{}.dense.weight", prefix))
                    .clone(),
                weights
                    .get_tensor(&format!("{}.dense.bias", prefix))
                    .clone(),
            ),
            layer_norm: LayerNorm::new(
                weights
                    .get_tensor(&format!("{}.LayerNorm.weight", prefix))
                    .clone(),
                weights
                    .get_tensor(&format!("{}.LayerNorm.bias", prefix))
                    .clone(),
                config.layer_norm_eps,
            ),
        }
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Tensor {
        let hidden_states = self.dense.forward(hidden_states);
        let hidden_states = hidden_states.add(input_tensor);
        self.layer_norm.forward(&hidden_states)
    }
}

impl BertIntermediate {
    fn new(weights: &ModelWeights, config: &ModelConfig, layer_idx: usize) -> Self {
        let prefix = format!("encoder.layer.{}.intermediate", layer_idx);

        Self {
            dense: Linear::new(
                weights
                    .get_tensor(&format!("{}.dense.weight", prefix))
                    .clone(),
                weights
                    .get_tensor(&format!("{}.dense.bias", prefix))
                    .clone(),
            ),
            intermediate_act_fn: ActivationFunction::from_str(&config.hidden_act),
        }
    }

    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let hidden_states = self.dense.forward(hidden_states);
        self.intermediate_act_fn.forward(&hidden_states)
    }
}

impl BertOutput {
    fn new(weights: &ModelWeights, config: &ModelConfig, layer_idx: usize) -> Self {
        let prefix = format!("encoder.layer.{}.output", layer_idx);

        Self {
            dense: Linear::new(
                weights
                    .get_tensor(&format!("{}.dense.weight", prefix))
                    .clone(),
                weights
                    .get_tensor(&format!("{}.dense.bias", prefix))
                    .clone(),
            ),
            layer_norm: LayerNorm::new(
                weights
                    .get_tensor(&format!("{}.LayerNorm.weight", prefix))
                    .clone(),
                weights
                    .get_tensor(&format!("{}.LayerNorm.bias", prefix))
                    .clone(),
                config.layer_norm_eps,
            ),
        }
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Tensor {
        let hidden_states = self.dense.forward(hidden_states);
        let hidden_states = hidden_states.add(input_tensor);
        self.layer_norm.forward(&hidden_states)
    }
}
