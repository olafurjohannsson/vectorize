use crate::models::ModelConfig;
use crate::tensor::Tensor;
use anyhow::Result;
use safetensors::SafeTensors;
use std::collections::HashMap;

pub struct ModelWeights {
    pub config: ModelConfig,
    pub tensors: HashMap<String, Tensor>,
}

impl ModelWeights {
    pub fn from_safetensors(path: &str) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("Reading file {} failed: {}", path, e))?;
        let tensors_data = SafeTensors::deserialize(&data)?;

        let mut tensors = HashMap::new();
        for (name, tensor_view) in tensors_data.tensors() {
            let shape = tensor_view.shape().to_vec();
            let data = tensor_view
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            tensors.insert(name.to_string(), Tensor::from_vec(data, shape));
        }

        let config_path = path.replace(".safetensors", "_config.json");

        let c = &std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Reading file {} failed: {}", config_path, e))?;

        let config: ModelConfig = serde_json::from_str(c)?;

        Ok(Self { config, tensors })
    }

    pub fn get_tensor(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .expect(&format!("Missing weight: {}", name))
    }
}
