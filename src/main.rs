use std::time::Instant;
use std::{error::Error, fs};

use env_logger::WriteStyle;
use log::info;
use nalgebra::DVector;
use rand::Rng;
use rand::{distributions::Alphanumeric, thread_rng};

use rand::seq::SliceRandom;
use crate::network::{cost_vectors, one_hot_vector};


mod data;
mod network;

const TRAINING_IMAGES_FILE: &str = "./data/train-images-idx3-ubyte";
const TRAINING_LABELS_FILE: &str = "./data/train-labels-idx1-ubyte";

fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    env_logger::builder()
        .write_style(WriteStyle::Always)
        .init();

    let training_data = data::read_data(TRAINING_IMAGES_FILE, TRAINING_LABELS_FILE)?;

    info!("Number of rows in training data: {}", training_data.labelled_images.len());
    info!("Image Dimensions: {:?}", training_data.image_dimensions);

    let input_size = training_data.image_dimensions.0 * training_data.image_dimensions.1;

    let layer_sizes = vec![input_size, 16, 16, 10];

    let network = network::NeuralNetwork::new_random(layer_sizes);

    info!("Input dimension: {}", network.input_dimension());
    info!("Output dimension: {}", network.output_dimension());

    let input = training_data.labelled_images.first().unwrap();
    let input_target_vec = one_hot_vector(input.label);
    let input_x = DVector::from_vec(input.image.iter().map(|int| *int as f32).collect());

    info!("Forward prop");
    let output = network.forward_propagation(&input_x);

    let cost = network::cost_digit(&output, training_data.labelled_images.first().unwrap().label);

    info!("cost: {cost}, output: {:?}", output.data.as_slice());
    
    info!("Backwards prop");
    let deltas = network.back_prop_single_input(&input_x, &one_hot_vector(input.label));
    
    for delta in &deltas {
        info!("delta: {:?}", delta.shape());
        info!("rank: {}", delta.rank(0.1));
        // info!("{}", delta);
    }

    info!("Shuffling inputs");
    let mut labelled_images = training_data.labelled_images;
    labelled_images.shuffle(&mut thread_rng());

    let backprop_inputs: Vec<_> = labelled_images.iter()
        .take(labelled_images.len() / 2)
        .map(|image| (
            DVector::from_vec(image.image.iter().map(|int| *int as f32).collect()),
            one_hot_vector(image.label)
        ))
        .collect();

    let save_prefix: String = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(4)
        .map(char::from)
        .collect();

    let save_prefix = format!("networks/{save_prefix}-gen");
    info!("Network will be saved with prefix {save_prefix:?}");
    
    info!("Starting training on {} inputs!", backprop_inputs.len());
    let mut network = network;
    for i in 0..100 {
        let path = format!("{save_prefix}{i}.json");
        info!("Saving network to {path:?}");
        let json = serde_json::to_string(&network);
        if let Ok(json) = json {
            let _ = fs::write(path, json);
        }
    
        let output = network.forward_propagation(&input_x);
        let cost = cost_vectors(&output, &input_target_vec);
        info!("[i={i}] cost for first image: {cost}");
    
        let start = Instant::now();
        network.backwards_propagation(&backprop_inputs);
        let duration = start.elapsed();
        info!("Generation took {}ms", duration.as_millis());
    }

    Ok(())
}
