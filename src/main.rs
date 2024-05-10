use std::error::Error;

use env_logger::WriteStyle;
use log::info;
use nalgebra::DVector;


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

    for matrix in &network.layer_weights {
        println!("Matrix dimensions: {:?}", matrix.shape())
    }

    let input = &training_data.labelled_images.first().unwrap().image;
    let input = DVector::from_vec(input.iter().map(|int| *int as f32).collect());

    let output = network.forward_propagation(input);
    let output = output.last().unwrap();

    let cost = network::cost_digit(output, training_data.labelled_images.first().unwrap().label);

    info!("cost: {cost}, output: {:?}", output.data.as_slice());

    Ok(())
}
