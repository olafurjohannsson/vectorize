use anyhow::Result;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {

    let cosine_sim = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    };

    println!("Starting vectorizer...");

    let vectorizer = vectorize_rs::Vectorizer::from_pretrained("minilm-l6-v2").await?;
    println!("Model loaded successfully!");

    let texts2 = vec!["physician", "doctor", "medical practice", "general"];
    let embeddings2 = vectorizer.encode(texts2.clone())?;

    println!(
        "\nLen: {}, First embedding (first 10 values): {:?}, Second embedding (first 10 values): {:?}",
        embeddings2.len(),
        &embeddings2[0][..10.min(embeddings2[0].len())],
        &embeddings2[1][..10.min(embeddings2[0].len())]
    ); // &embeddings[0][..10.min(embeddings[0].len())]
    println!("\nCosine similarities:");
    for i in 0..texts2.len() {
        for j in i + 1..texts2.len() {
            let sim = cosine_sim(&embeddings2[i], &embeddings2[j]);
            println!("'{}' vs '{}': {:.3}", texts2[i], texts2[j], sim);
        }
    }


    let texts = vec!["Hello world", "How are you?"];
    println!("Encoding texts: {:?}", texts);

    let embeddings = vectorizer.encode(texts.clone())?;
    println!(
        "Embeddings shape: [{}, {}]",
        embeddings.len(),
        embeddings[0].len()
    );



    println!("\nCosine similarities:");
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            let sim = cosine_sim(&embeddings[i], &embeddings[j]);
            println!("'{}' vs '{}': {:.3}", texts[i], texts[j], sim);
        }
    }

    println!(
        "\nFirst embedding (first 10 values): {:?}",
        &embeddings[0][..10.min(embeddings[0].len())]
    );

    let first_emb = &embeddings[0];
    let mean: f32 = first_emb.iter().sum::<f32>() / first_emb.len() as f32;
    let variance: f32 =
        first_emb.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / first_emb.len() as f32;
    let std = variance.sqrt();
    let min = first_emb.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = first_emb.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let norm = first_emb.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\nStatistics for first embedding:");
    println!("Mean: {}", mean);
    println!("Std: {}", std);
    println!("Min: {}", min);
    println!("Max: {}", max);
    println!("L2 Norm: {}", norm);

    Ok(())
}
