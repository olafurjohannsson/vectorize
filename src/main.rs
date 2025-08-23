
use tokio;
use anyhow;
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let vectorizer = vectorize_rs::Vectorizer::from_pretrained("minilm-l6-v2").await?;
    let embeddings = vectorizer.encode(vec!["Hello world", "How are you?"])?;

    println!("Embedding shape: [{}, {}]", embeddings.len(), embeddings[0].len());
    println!("First embedding: {:?}", &embeddings[0][..5]); // First 5 values

    Ok(())
}