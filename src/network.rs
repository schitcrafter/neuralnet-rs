use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

#[derive(Clone)]
pub struct NeuralNetwork {
    pub layer_sizes: Vec<u32>,
    pub layer_weights: Vec<DMatrix<f32>>,
    pub layer_biases: Vec<DVector<f32>>,
}

impl NeuralNetwork {
    pub fn new_random(layer_sizes: Vec<u32>) -> NeuralNetwork {
        if layer_sizes.len() < 2 {
            panic!("Can't have less than two layers");
        }

        let mut layer_weights = Vec::new();

        let mut rng = rand::thread_rng();

        for (left, right) in layer_sizes.iter().tuple_windows() {
            let new_matrix = DMatrix::from_fn(*right as usize, *left as usize, |_, _| {
                rng.gen_range(-1.0..=1.0)
            });
            layer_weights.push(new_matrix);
        }

        assert_eq!(layer_sizes.len(), layer_weights.len() + 1);
        assert_eq!(layer_weights[0].nrows(), layer_weights[1].ncols());

        let mut layer_biases = Vec::new();

        for layer_size in &layer_sizes[1..] {
            let new_vector = DVector::from_fn(*layer_size as usize,
                |_, _| rng.gen_range(-1.0..=0.0));
            
            layer_biases.push(new_vector);
        }

        assert_eq!(layer_biases.len(), layer_sizes.len() - 1);

        NeuralNetwork {
            layer_sizes,
            layer_weights,
            layer_biases
        }
    }

    /// Returns the intermediate states of all neurons.
    /// The first element will be the input, the last element the output.
    pub fn forward_propagation(&self, input: DVector<f32>) -> Vec<DVector<f32>> {
        let mut intermediate_outputs = vec![input];

        for (weights, biases) in self.layer_weights.iter().zip(&self.layer_biases) {
            let input = intermediate_outputs.last().unwrap();

            let mut output = weights * input;
            output += biases;
            output.apply(|value| *value = sigmoid(*value));

            intermediate_outputs.push(output);
        }

        intermediate_outputs
    }

    pub fn back_prop(&mut self, _inputs: Vec<DVector<f32>>) {

    }

    /// Does backpropagation for a single input, returning
    /// the wanted changes to the weights.
    pub fn back_prop_single_input(&self, _input: DVector<f32>) -> Vec<DMatrix<f32>> {
        vec![]
    }

    pub fn input_dimension(&self) -> u32 {
        *self.layer_sizes.first().unwrap()
    }

    pub fn output_dimension(&self) -> u32 {
        *self.layer_sizes.last().unwrap()
    }
}

pub fn cost_digit(output: &DVector<f32>, digit: u8) -> f32 {
    let expected = expected_output(digit);
    cost_vectors(output, &expected)
}

/// Calculates the sum of the squared differences for the cost.
pub fn cost_vectors(output: &DVector<f32>, expected_output: &DVector<f32>) -> f32 {
    let sum: f32 = output.data.as_slice().iter()
        .zip(expected_output.data.as_slice())
        .map(|(actual, expected)| (actual - expected) * (actual - expected))
        .sum();

    sum * 0.5
}

pub fn expected_output(digit: u8) -> DVector<f32> {
    let mut vec = vec![0f32; 10];
    vec[digit as usize] = 1.0;

    DVector::from_vec(vec)
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-z))
}
