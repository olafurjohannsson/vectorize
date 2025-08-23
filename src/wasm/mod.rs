use wasm_bindgen::prelude::*;
use crate::Vectorizer;

#[wasm_bindgen]
pub struct WasmVectorizer {
    inner: Vectorizer,
}

#[wasm_bindgen]
impl WasmVectorizer {
    #[wasm_bindgen(constructor)]
    pub async fn new(model_name: String) -> Result<WasmVectorizer, JsValue> {
        let inner = Vectorizer::from_pretrained(&model_name)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen]
    pub fn encode(&self, texts: Vec<String>) -> Result<Vec<f32>, JsValue> {
        let texts_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.inner.encode(texts_refs)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // JS flattening
        Ok(embeddings.into_iter().flatten().collect())
    }
}