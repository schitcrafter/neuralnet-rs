use itertools::Itertools;
use log::debug;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const DEFAULT_LEARNING_RATE: f32 = 10.0;

#[derive(Clone, Deserialize, Serialize)]
pub struct NeuralNetwork {
    pub layer_sizes: Vec<u32>,
    pub layer_weights: Vec<DMatrix<f32>>,
    pub learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new_random(layer_sizes: Vec<u32>) -> NeuralNetwork {
        if layer_sizes.len() < 2 {
            panic!("Can't have less than two layers");
        }

        let mut layer_weights = Vec::new();

        let mut rng = rand::thread_rng();

        for (left, right) in layer_sizes.iter().tuple_windows() {
            let new_matrix = DMatrix::from_fn(*right as usize, *left as usize + 1, |_, _| {
                rng.gen_range(-1.0..=1.0)
            });
            layer_weights.push(new_matrix);
        }

        assert_eq!(layer_sizes.len(), layer_weights.len() + 1);
        if layer_weights.len() >= 2 {
            assert_eq!(layer_weights[0].nrows(), layer_weights[1].ncols() - 1);
        }

        NeuralNetwork {
            layer_sizes,
            layer_weights,
            learning_rate: DEFAULT_LEARNING_RATE
        }
    }

    /// Returns the activation of all neurons, plus the derivative of the activation
    /// function evaluated with the activation.
    /// The first element will be the input, the last element the output.
    pub fn forward_propagation(&self, input: &DVector<f32>) -> DVector<f32> {
        let mut last_output = input.clone();

        for weights in self.layer_weights.iter() {
            let mut activation = weights * last_output.push(1.0);

            activation.apply(|value| *value = sigmoid(*value));

            last_output = activation;
        }

        last_output
    }

    /// returns the average cost of each datapoint,
    /// as well as which percentage was correctly classified
    pub fn average_cost_correct_classified(&self, inputs: &[(DVector<f32>, u8)]) -> (f32, f32) {
        let (added_costs, correct_classified): (f32, usize) = inputs.par_iter()
            .map(|(input, expected)| {
                let output = self.forward_propagation(input);
                let cost = cost_digit(&output, *expected);
                let correctly_classified = correctly_classified(&output, *expected);
                (cost, correctly_classified as usize)
            })
            .reduce(|| (0.0, 0),
                |acc, x| (acc.0 + x.0, acc.1 + x.1)
            );
        
        let correct_classified_percent = (correct_classified as f32) * 100.0 / inputs.len() as f32;
        (added_costs / inputs.len() as f32, correct_classified_percent)
    }
    
    /// Does backpropagation for a single input, returning
    /// the wanted changes to the weights.
    pub fn back_prop_single_input(&self, input: &DVector<f32>, y_digit: u8) -> Vec<DMatrix<f32>> {
        let y = one_hot_vector(y_digit);
        // do forward prop and get all information out of it that we can
        let mut activations = vec![input.clone()];
        
        for weights in self.layer_weights.iter() {
            let last_output = activations.last().unwrap();
            let mut activation = weights * last_output.push(1.0);

            activation.apply(|value| *value = sigmoid(*value));

            activations.push(activation);
        }

        // This will store the gradients all weights will go to, right to left
        let mut weight_gradients = vec![];

        let output_activ = activations.last().unwrap();
        let error_deriv = output_activ - y;
        let mut last_delta = error_deriv.component_mul(&output_activ.map(sigmoid_derivative));

        let weight_activation_iter =
            self.layer_weights.iter().rev()
            .zip(activations.into_iter().rev().tuple_windows());

        for (weights, (activation_right, activation_left)) in weight_activation_iter {
            debug!("weights: {:?}, activation_left: {:?}, activation_right: {:?}", weights.shape(), activation_left.shape(), activation_right.shape());

            let weight_gradient = &last_delta * activation_left.push(1.0).transpose();
            assert_eq!(weight_gradient.shape(), weights.shape(), "Shape of weight gradient matrix is wrong");
            weight_gradients.push(weight_gradient);
            
            let weights_transpose_no_bias = weights.columns(0, weights.ncols()-1).transpose();
            let new_delta = weights_transpose_no_bias * last_delta;
            let activation_deriv = activation_left.map(sigmoid_derivative);

            let new_delta = activation_deriv.component_mul(&new_delta);

            last_delta = new_delta;
        }

        weight_gradients.reverse();

        assert_eq!(weight_gradients.len(), self.layer_weights.len());
        for (weight_gradient, weight) in weight_gradients.iter().zip(&self.layer_weights) {
            assert_eq!(weight_gradient.shape(), weight.shape());
        }

        weight_gradients
    }

    /// inputs is a list of input vectors to expected outputs
    pub fn backwards_propagation(&mut self, inputs: &[(DVector<f32>, u8)]) {
        let mut changes: Vec<DMatrix<f32>> = Vec::new();

        for weight_mat in &self.layer_weights {
            let zero_mat = DMatrix::zeros(weight_mat.nrows(), weight_mat.ncols());
            changes.push(zero_mat);
        }

        let mut zero_matrices = Vec::new();
        for weight in &self.layer_weights {
            zero_matrices.push(DMatrix::zeros(weight.nrows(), weight.ncols()));
        }

        let added_deltas = inputs.par_iter()
            .map(|(input, target)|
                self.back_prop_single_input(input, *target))
            .reduce(|| zero_matrices.clone(), |acc, deltas| {
                acc.into_iter().zip(deltas)
                    .map(|(l, r)| l + r)
                    .collect()
            });

        let input_length = inputs.len() as f32;
        let added_deltas: Vec<_> = added_deltas.into_iter()
            .map(|weight_delta| (weight_delta * self.learning_rate) / input_length)
            .collect();

        self.layer_weights = self.layer_weights.iter()
            .zip(added_deltas)
            .map(|(weights, deltas)| weights - deltas)
            .collect();
        
    }

    pub fn input_dimension(&self) -> u32 {
        *self.layer_sizes.first().unwrap()
    }

    pub fn output_dimension(&self) -> u32 {
        *self.layer_sizes.last().unwrap()
    }
}

pub fn cost_digit(output: &DVector<f32>, digit: u8) -> f32 {
    let expected = one_hot_vector(digit);
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

pub fn one_hot_vector(digit: u8) -> DVector<f32> {
    let mut vec = vec![0f32; 10];
    vec[digit as usize] = 1.0;

    DVector::from_vec(vec)
}

fn correctly_classified(output: &DVector<f32>, digit: u8) -> bool {
    output.data.as_slice().iter().position_max_by(
        |x, y| x.total_cmp(y)
    ) == Some(digit as usize)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-x))
}

/// takes the derivative of the sigmoid, taking the output
/// of the sigmoid as output.
fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_random_network() {
        let net = NeuralNetwork::new_random(vec![10, 10]);
        
        assert_eq!(net.layer_weights.len(), 1);
        let input = DVector::from_vec(
            vec![1.0; 10]
        );
        let output = net.forward_propagation(&input);
        assert_eq!(output.shape(), (10, 1));
    }
}
