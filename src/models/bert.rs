use crate::tensor::Tensor;
use crate::models::ModelConfig;
use crate::weights::ModelWeights;
use crate::models::components::{ActivationFunction, Linear, Embedding, LayerNorm};

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
        // get embeddings
        let embedding_output = self.embeddings.forward(input_ids, None);

        // encoder uass through
        let encoder_output = self.encoder.forward(embedding_output, Some(attention_mask));

        // mean pooling for sentence embeddings
        self.mean_pooling(encoder_output, attention_mask)
    }

    fn mean_pooling(&self, hidden_states: Tensor, attention_mask: &Tensor) -> Tensor {
        // mask padding tokens
        let input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(&hidden_states);
        let sum_embeddings = hidden_states.mul(&input_mask_expanded).sum(&[-1], false);
        let sum_mask = input_mask_expanded.sum(&[-1], false).clamp_min(1e-9);
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
                weights.get_tensor("embeddings.word_embeddings.weight").clone(),
            ),
            position_embeddings: Embedding::new(
                weights.get_tensor("embeddings.position_embeddings.weight").clone(),
            ),
            token_type_embeddings: Embedding::new(
                weights.get_tensor("embeddings.token_type_embeddings.weight").clone(),
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
        let seq_length = input_ids.shape()[1];

        // word embeddings
        let inputs_embeds = self.word_embeddings.forward(input_ids);

        // position IDs [0, 1, 2, ..., seq_length-1]
        let position_ids = Tensor::arange(0, seq_length as i64);
        let position_embeddings = self.position_embeddings.forward(&position_ids);

        let zeros_tensor;
        let token_type_ids = if let Some(ids) = token_type_ids {
            ids
        } else {
            zeros_tensor = Tensor::zeros_like(input_ids);
            &zeros_tensor
        };
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);

        let embeddings = inputs_embeds.add(&position_embeddings).add(&token_type_embeddings);

        // apply layer norm
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
        let attention_output = self.attention.self_attention.forward(hidden_states, attention_mask);
        let attention_output = self.attention.output.forward(&attention_output, hidden_states);

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
                weights.get_tensor(&format!("{}.query.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.query.bias", prefix)).clone(),
            ),
            key: Linear::new(
                weights.get_tensor(&format!("{}.key.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.key.bias", prefix)).clone(),
            ),
            value: Linear::new(
                weights.get_tensor(&format!("{}.value.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.value.bias", prefix)).clone(),
            ),
            num_attention_heads,
            attention_head_size,
        }
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {

        let query_layer = self.transpose_for_scores(&self.query.forward(hidden_states));
        let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states));
        let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states));

        // attention
        let attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2));
        let attention_scores = attention_scores.div_scalar((self.attention_head_size as f32).sqrt());

        // Apply attention mask if provided
        let attention_scores = if let Some(mask) = attention_mask {
            attention_scores.add(&mask.mul_scalar(-10000.0))
        } else {
            attention_scores
        };

        // normalize softmax
        let attention_probs = attention_scores.softmax(-1);
        // apply attention
        let context_layer = attention_probs.matmul(&value_layer);
        // Reshape
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

        x.transpose(-2, -3)
            .reshape(&[batch_size, seq_len, self.num_attention_heads * self.attention_head_size])
    }
}

impl BertSelfOutput {
    fn new(weights: &ModelWeights, config: &ModelConfig, layer_idx: usize) -> Self {
        let prefix = format!("encoder.layer.{}.attention.output", layer_idx);

        Self {
            dense: Linear::new(
                weights.get_tensor(&format!("{}.dense.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.dense.bias", prefix)).clone(),
            ),
            layer_norm: LayerNorm::new(
                weights.get_tensor(&format!("{}.LayerNorm.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.LayerNorm.bias", prefix)).clone(),
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
                weights.get_tensor(&format!("{}.dense.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.dense.bias", prefix)).clone(),
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
                weights.get_tensor(&format!("{}.dense.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.dense.bias", prefix)).clone(),
            ),
            layer_norm: LayerNorm::new(
                weights.get_tensor(&format!("{}.LayerNorm.weight", prefix)).clone(),
                weights.get_tensor(&format!("{}.LayerNorm.bias", prefix)).clone(),
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