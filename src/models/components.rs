use crate::tensor::Tensor;

pub struct Embedding {
    weight: Tensor, // [vocab_size, hidden_size]
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        // Gather embeddings for each input ID
        self.weight.gather(0, input_ids)
    }
}

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias: Some(bias) }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weight.transpose(-1, -2));
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            output
        }
    }
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        // Calculate mean and variance
        let mean = hidden_states.mean(&[-1], true);
        let variance = hidden_states.sub(&mean).pow(2.0).mean(&[-1], true);

        // Normalize
        let hidden_states = hidden_states.sub(&mean).div(&variance.add_scalar(self.eps).sqrt());

        // Scale and shift
        hidden_states.mul(&self.weight).add(&self.bias)
    }
}

pub enum ActivationFunction {
    Gelu,
    Relu,
    Tanh,
}

impl ActivationFunction {
    pub fn from_str(s: &str) -> Self {
        match s {
            "gelu" | "gelu_new" => Self::Gelu,
            "relu" => Self::Relu,
            "tanh" => Self::Tanh,
            _ => Self::Gelu, // Default
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Self::Gelu => {
                // GELU approximation used by BERT
                // x * 0.5 * (1.0 + tanh(sqrt(2.0 / π) * (x + 0.044715 * x^3)))
                let inner = x.mul_scalar(0.7978845608)
                    .add(&x.pow(3.0).mul_scalar(0.044715))
                    .tanh();
                x.mul_scalar(0.5).mul(&inner.add_scalar(1.0))
            },
            Self::Relu => x.relu(),
            Self::Tanh => x.tanh(),
        }
    }
}