use anyhow::Result;
use tokenizers::Tokenizer;

pub mod models;
pub mod tensor;
pub mod weights;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

use models::{EmbeddingModel, ModelFactory};
use tensor::Tensor;
use weights::ModelWeights;

pub struct Vectorizer {
    model: Box<dyn EmbeddingModel>,
    tokenizer: Tokenizer,
}

impl Vectorizer {
    pub async fn from_pretrained(model_name: &str) -> Result<Self> {
        let weights = ModelWeights::from_safetensors(&format!("{}.safetensors", model_name))?;

        let tokenizer_file = &format!("{}_tokenizer.json", model_name);
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load {} tokenizer: {}", tokenizer_file, e))?;

        let model_type = weights.config.model_type.clone();
        let model = ModelFactory::create(&model_type, weights);

        Ok(Self { model, tokenizer })
    }

    pub fn encode(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // get tensors and embedding
        let (input_ids, attention_mask) = self.prepare_inputs(encodings);
        let embeddings = self.model.encode(&input_ids, &attention_mask);

        Ok(embeddings.to_vec_2d())
    }

    fn prepare_inputs(&self, encodings: Vec<tokenizers::Encoding>) -> (Tensor, Tensor) {
        let batch_size = encodings.len();
        if batch_size == 0 {
            return (Tensor::zeros(vec![0, 0]), Tensor::zeros(vec![0, 0]));
        }


        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        println!("Batch size: {}, Max length: {}", batch_size, max_len);

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            println!("Text {}: {} tokens, first 10 IDs: {:?}",
                     i, ids.len(), &ids[..ids.len().min(10)]);
        }
        let mut input_ids_data = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_data = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            // Add the actual ids and mask values
            for i in 0..max_len {
                if i < ids.len() {
                    input_ids_data.push(ids[i] as f32);
                    attention_mask_data.push(mask[i] as f32);
                } else {
                    // Padding
                    input_ids_data.push(0.0);
                    attention_mask_data.push(0.0);
                }
            }
        }

        // Create tensors with shape [batch_size, max_len]
        let input_ids = Tensor::from_vec(input_ids_data, vec![batch_size, max_len]);
        let attention_mask = Tensor::from_vec(attention_mask_data, vec![batch_size, max_len]);

        (input_ids, attention_mask)
    }
}
