pub mod bert;
pub mod components;

use crate::tensor::Tensor;
use crate::weights::ModelWeights;

pub trait EmbeddingModel: Send + Sync {
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Tensor;
    fn get_config(&self) -> &ModelConfig;
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
}

pub struct ModelFactory;

impl ModelFactory {
    pub fn create(model_type: &str, weights: ModelWeights) -> Box<dyn EmbeddingModel> {
        match model_type {
            "bert" | "minilm" => Box::new(bert::BertModel::new(weights)),
            _ => panic!("Unsupported model type: {}", model_type),
        }
    }
}