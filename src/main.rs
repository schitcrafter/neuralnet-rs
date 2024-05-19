use std::time::Instant;
use std::error::Error;

use env_logger::WriteStyle;
use log::info;
use nalgebra::DVector;
use rand::Rng;
use rand::{distributions::Alphanumeric, thread_rng};

use rand::seq::SliceRandom;

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
    let input_x = DVector::from_vec(input.image.iter().map(|int| *int as f32).collect());

    info!("Forward prop");
    let output = network.forward_propagation(&input_x);

    let cost = network::cost_digit(&output, training_data.labelled_images.first().unwrap().label);

    info!("cost: {cost}, output: {:?}", output.data.as_slice());
    
    info!("Backwards prop");
    let _deltas = network.back_prop_single_input(&input_x, input.label);
    
    let mut labelled_images = training_data.labelled_images;
    labelled_images.shuffle(&mut thread_rng());

    let backprop_inputs: Vec<_> = labelled_images.iter()
        .map(|image| (
            DVector::from_vec(image.image.iter().map(|int| *int as f32).collect()),
            image.label
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

    let mut classified_percent_history = vec![];
    let mut last_cost = 0.0;
    for i in 0..1000 {
        // let path = format!("{save_prefix}{i}.json");
        // info!("Saving network to {path:?}");
        // let json = serde_json::to_string(&network);
        // if let Ok(json) = json {
        //     let _ = fs::write(path, json);
        // }
        
        let start = Instant::now();

        network.backwards_propagation(&backprop_inputs);

        let duration = start.elapsed();
        if i % 10 == 0 {
            let (cost, correct_classified) = network.average_cost_correct_classified(&backprop_inputs);

            let delta_classified = correct_classified - classified_percent_history.last().unwrap_or(&0.0);
            let delta_cost = cost - last_cost;

            last_cost = cost;
            classified_percent_history.push(correct_classified);

            info!("[gen {i}] took {}ms, {correct_classified:.2}% correctly classified (delta: {delta_classified:.2}), average cost: {cost:.4} (delta: {delta_cost:.4})",
                duration.as_millis());
        }
    }

    Ok(())
}
