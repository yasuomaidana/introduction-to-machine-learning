mod args;
mod utils;
// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;
//
// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use crate::args::Args;
use anyhow::{Error as E, Result};
use candle_core::Tensor;
use clap::Parser;
use tokenizers::PaddingParams;

fn main() {
    let args = Args::parse();
    println!("{:?}", args);
    let (model, mut tokenizer) = args.build_model_and_tokenizer().unwrap();

    let start = std::time::Instant::now();

    let device = &model.device;

    if let Some(prompt) = args.prompt {
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)
            .unwrap();
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let token_type_ids = token_ids.zeros_like().unwrap();
        println!("Loaded and encoded {:?}", start.elapsed());
        for idx in 0..args.n {
            let start = std::time::Instant::now();
            let ys = model.forward(&token_ids, &token_type_ids, None).unwrap();
            if idx == 0 {
                println!("{ys}");
            }
            println!("Took {:?}", start.elapsed());
        }
    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        let n_sentences = sentences.len();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)
            .unwrap();
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device).expect("failed to construct tokenizer"))
            })
            .collect::<Result<Vec<_>>>()
            .expect("failed to construct tokenizer");
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device).expect("failed to construct tokenizer"))
            })
            .collect::<Result<Vec<_>>>()
            .expect("failed to construct tokenizer");

        let token_ids = Tensor::stack(&token_ids, 0).unwrap();
        let attention_mask = Tensor::stack(&attention_mask, 0).unwrap();
        let token_type_ids = token_ids.zeros_like().unwrap();
        println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))
            .unwrap();
        println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();
        let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();
        let embeddings = if args.normalize_embeddings {
            normalize_l2(&embeddings).unwrap()
        } else {
            embeddings
        };
        println!("pooled embeddings {:?}", embeddings.shape());

        let mut similarities = vec![];
        for i in 0..n_sentences {
            let e_i = embeddings.get(i).unwrap();
            for j in (i + 1)..n_sentences {
                let e_j = embeddings.get(j).unwrap();
                let sum_ij = (&e_i * &e_j)
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                let sum_i2 = (&e_i * &e_i)
                    .expect("Failed to compute e_i squared")
                    .sum_all()
                    .expect("Failed to sum e_i squared")
                    .to_scalar::<f32>()
                    .expect("Failed to convert e_i squared to scalar");
                let sum_j2 = (&e_j * &e_j)
                    .expect("Failed to compute e_j squared")
                    .sum_all()
                    .expect("Failed to sum e_j squared")
                    .to_scalar::<f32>()
                    .expect("Failed to convert e_j squared to scalar");
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j))
            }
        }
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
