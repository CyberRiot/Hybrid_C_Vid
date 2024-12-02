#include "../include/layer.hpp"
#include <chrono>
#include <algorithm>

//Constructor: initializing filters with random values
ConvLayer::ConvLayer(int filter_size, int num_filters, double learning_rate)
    : filter_size(filter_size), num_filters(num_filters), learning_rate(learning_rate), convolution_done(false) {
    filters = new std::vector<std::vector<neuron*>>(num_filters);
    conv_output = nullptr;  // Initialize to null to avoid dangling pointers

    for (int i = 0; i < num_filters; i++) {
        (*filters)[i] = std::vector<neuron*>(filter_size * filter_size);
        for (int j = 0; j < filter_size * filter_size; j++) {
            (*filters)[i][j] = new neuron(filter_size * filter_size);
        }
    }
}

ConvLayer::~ConvLayer() {
    // Clean up the filters
    for (auto& filter : *filters) {
        for (neuron* n : filter) {
            delete n;
        }
    }
    delete filters;
}

// Forward Pass: Apply convolution on the input
std::vector<double>* ConvLayer::forward(std::vector<double>* input) {
    if (input == nullptr || input->empty()) {
        throw std::runtime_error("Error: ConvLayer received an empty input vector!");
    }

    // Check input size for squareness
    int image_size = static_cast<int>(std::sqrt(input->size()));
    if (image_size * image_size != input->size()) {
        throw std::runtime_error("Error: ConvLayer received a non-square input vector!");
    }

    // Reshape the input
    auto reshaped_input = new std::vector<std::vector<double>>(image_size, std::vector<double>(image_size));
    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < image_size; ++j) {
            (*reshaped_input)[i][j] = (*input)[i * image_size + j];
        }
    }

    // Perform convolution
    auto convolved_output = new std::vector<double>();
    for (int i = 0; i < num_filters; ++i) {
        auto filter_output = convolve(reshaped_input, &((*filters)[i]));
        for (auto& row : *filter_output) {
            convolved_output->insert(convolved_output->end(), row.begin(), row.end());
        }
        delete filter_output;
    }
    delete reshaped_input;

    // Apply pooling
    auto pooled_output = average_pooling(convolved_output, 2);  // Example: pooling size of 2
    delete convolved_output;

    if (pooled_output == nullptr) {
        throw std::runtime_error("Error: pooled_output is null after pooling!");
    }

    if (pooled_output->size() == 0) {
        throw std::runtime_error("Error: pooled_output is empty after pooling!");
    }

    std::cout << "Convolution and pooling completed. Output size: " << pooled_output->size() << std::endl;
    return pooled_output;  // Return the processed output
}

std::vector<double>* ConvLayer::backward(std::vector<double>* gradients) {
    // Assuming d_out is the gradient of the loss w.r.t the output of this layer
    std::vector<double>* d_input = new std::vector<double>(input->size());  // Gradient w.r.t input
    std::vector<std::vector<double>> d_filters(num_filters, std::vector<double>(filter_size * filter_size)); // Gradient w.r.t filters

    // Loop over the filters and inputs to compute gradients
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < input->size(); ++i) {
            for (int j = 0; j < filter_size * filter_size; ++j) {
                d_filters[f][j] += (*input)[i] * (*gradients)[i];  // Adjust the filters using chain rule
                (*d_input)[i] += (*filters)[f][j]->weights->at(j) * (*gradients)[i];  // Propagate gradient back to input
            }
        }
    }

    // Update the filters using gradients (gradient descent)
    for (int f = 0; f < num_filters; ++f) {
        for (int j = 0; j < filter_size * filter_size; ++j) {
            for (auto &n : (*filters)[f]) {
                n->weights->at(j) -= learning_rate * d_filters[f][j];  // Update weights using the learning rate
            }
        }
    }

    return d_input;  // Return the gradient w.r.t input
}

std::vector<std::vector<double>>* ConvLayer::convolve(std::vector<std::vector<double>>* input, std::vector<neuron*>* filter) {
    int input_size = input->size();  // Input size should match the reshaped 2D matrix
    int output_size = input_size - filter_size + 1;  // Calculate output size

    std::cout << "Starting convolution process..." << std::endl;
    std::cout << "Input size: " << input_size << ", Filter size: " << filter_size 
              << ", Output size: " << output_size << std::endl;
    std::cout << "Filter neuron count: " << filter->size() << std::endl;

    // Check for valid output size before proceeding
    if (output_size <= 0) {
        std::cerr << "Error: Output size is zero or negative. Input size: " << input_size 
                  << ", Filter size: " << filter_size 
                  << ", Calculated output size: " << output_size 
                  << ". Skipping this convolution layer." << std::endl;
        return nullptr;  // Skip the convolution if output size is invalid
    }

    // Proceed with convolution if sizes are valid
    auto result = new std::vector<std::vector<double>>(output_size, std::vector<double>(output_size));

    // Perform 2D convolution
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double convolved_value = 0;
            int filter_idx = 0;
            for (int fi = 0; fi < filter_size; fi++) {
                for (int fj = 0; fj < filter_size; fj++) {
                    if (filter_idx >= filter->size()) {
                        std::cerr << "Error: Filter neuron out of bounds at index " << filter_idx << std::endl;
                        return nullptr;
                    }

                    neuron* n = (*filter)[filter_idx++];
                    if (!n) {
                        std::cerr << "Error: Null neuron at filter index " << filter_idx - 1 << std::endl;
                        return nullptr;
                    }

                    double input_value = (*input)[i + fi][j + fj];
                    double weight_value = n->weights->at(fi * filter_size + fj);

                    convolved_value += input_value * weight_value;
                }
            }
            (*result)[i][j] = convolved_value;
        }
    }

    std::cout << "\nConvolution process completed!" << std::endl;
    return result;
}

std::vector<double>* ConvLayer::average_pooling(std::vector<double>* input, int pooling_size) {
    int input_size = static_cast<int>(std::sqrt(input->size()));
    int output_size = input_size / pooling_size;
    auto output = new std::vector<double>(output_size * output_size, 0.0);

    std::cout << "Pooling input size: " << input->size() << " | Pooling output size: " << output->size() << std::endl;

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double sum = 0.0;
            for (int x = 0; x < pooling_size; x++) {
                for (int y = 0; y < pooling_size; y++) {
                    sum += (*input)[(i * pooling_size + x) * input_size + (j * pooling_size + y)];
                }
            }
            (*output)[i * output_size + j] = sum / (pooling_size * pooling_size);
        }
    }

    // Log pooled data for inspection
    return output;
}

int ConvLayer::get_pooled_output_size() const {
    return this->input->size();  // Return the size after pooling
}

// RNNLayer (LSTM/GRU) Implementation

//Initialize the hidden and cell states for usage
RNNLayer::RNNLayer(int input_size_param, int hidden_size_param, double learning_rate_param){
    input_size = input_size_param;
    hidden_size = hidden_size_param;
    this->learning_rate = learning_rate;

    hidden_state = new std::vector<double>(hidden_size, 0.0);
    cell_state = new std::vector<double>(hidden_size, 0.0);
    hidden_neurons = new std::vector<neuron*>(hidden_size);

    // Initialize hidden_neurons with neurons of appropriate input size
    for (int i = 0; i < hidden_size; i++) {
        (*hidden_neurons)[i] = new neuron(input_size);  // Ensure neurons are initialized with the correct input size
    }
}

RNNLayer::~RNNLayer() {
    delete hidden_state;
    delete cell_state;
    for (neuron* n : *hidden_neurons) {
        delete n;
    }
    delete hidden_neurons;
}

// Inside the RNNLayer's forward pass method
std::vector<double>* RNNLayer::forward(std::vector<double>* input) {
    if (input == nullptr || input->empty()) {
        throw std::runtime_error("Invalid input to RNNLayer::forward. Input is null or empty.");
    }

    std::cout << "Starting RNNLayer forward pass with input size: " << input->size() << std::endl;

    int max_chunks = 50;  // Limit the number of chunks
    int chunk_size = std::max(1, static_cast<int>(input->size() / max_chunks));
    int calculated_chunks = (input->size() + chunk_size - 1) / chunk_size;

    std::cout << "Chunk Size: " << chunk_size << ", Number of Chunks: " << calculated_chunks << std::endl;

    // Process input data in chunks
    std::vector<double>* rnn_output = new std::vector<double>();
    for (int chunk_index = 0; chunk_index < calculated_chunks; ++chunk_index) {
        int chunk_start = chunk_index * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(input->size()));

        // Validate chunk indices
        if (chunk_start >= input->size()) {
            throw std::runtime_error("Invalid chunk start index.");
        }
        if (chunk_end <= chunk_start) {
            throw std::runtime_error("Invalid chunk size detected.");
        }

        std::cout << "Processing chunk " << chunk_index + 1 << "/" << calculated_chunks
                  << " with start: " << chunk_start << " and end: " << chunk_end << std::endl;

        // Extract the chunk and process it
        std::vector<double> chunk(input->begin() + chunk_start, input->begin() + chunk_end);
        std::vector<double>* chunk_output = this->forward_chunk(&chunk);

        // Append the chunk output to the overall RNN output
        rnn_output->insert(rnn_output->end(), chunk_output->begin(), chunk_output->end());
        delete chunk_output;
    }

    std::cout << "RNNLayer forward pass completed. Output size: " << rnn_output->size() << std::endl;
    return rnn_output;
}

std::vector<double>* RNNLayer::forward_chunk(std::vector<double>* chunk) {
    if (chunk == nullptr || chunk->empty()) {
        throw std::runtime_error("Invalid chunk input. Chunk is null or empty.");
    }

    std::cout << "Processing RNN chunk of size: " << chunk->size() << std::endl;

    // Placeholder for RNN forward logic
    // You can replace this with the actual computation for your RNN layer
    std::vector<double>* output = new std::vector<double>(chunk->size(), 0.0);  // Dummy output
    for (size_t i = 0; i < chunk->size(); ++i) {
        // Example: Copy input directly to output for now
        (*output)[i] = (*chunk)[i] * 0.5;  // Dummy operation
    }

    std::cout << "Chunk processed. Output size: " << output->size() << std::endl;
    return output;
}


std::vector<double>* RNNLayer::backward(std::vector<double>* gradients) {
    int num_chunks = cd->get_num_chunks();  // Get num_chunks from common_data
    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);

    std::cout << "Entered RNNLayer backward pass" << std::endl;

    // Process each chunk separately
    for (int chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
        int chunk_start = chunk_index * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(gradients->size()));

        // Create chunk for the current part of the gradient
        std::vector<double> chunk(gradients->begin() + chunk_start, gradients->begin() + chunk_end);

        std::cout << "Processing chunk " << chunk_index + 1 << "/" << num_chunks << std::endl;

        // Process the gradients for this chunk
        for (int i = 0; i < hidden_size; ++i) {
            // Gradient for each hidden neuron
            double error = chunk[i];  // Error for this chunk and hidden neuron

            // Propagate the error through the hidden state
            // Here you are calculating the error gradient for this particular layer
            // You would multiply by the derivative of your activation function if you were applying one.
            (*d_input)[i] = error;  // Simply propagate the error in this case
        }
    }

    return d_input;
}



std::vector<double>* RNNLayer::lstm_forward(std::vector<double>* input) {
    std::cout << "Entered LSTM_Forward" << std::endl;
    auto forget_gate = new std::vector<double>(hidden_size);
    auto input_gate = new std::vector<double>(hidden_size);
    auto output_gate = new std::vector<double>(hidden_size);
    auto cell_candidate = new std::vector<double>(hidden_size);

    // Output for the current time step
    auto output = new std::vector<double>(hidden_size);

    // Debug outputs for sizes
    std::cout << "Input size: " << input->size() 
              << ", Hidden size: " << hidden_size 
              << ", Hidden neurons size: " << hidden_neurons->size() 
              << std::endl;

    if (hidden_neurons->size() != hidden_size) {
        std::cerr << "Error: Hidden neurons size mismatch. Expected: " << hidden_size 
                  << ", Got: " << hidden_neurons->size() << std::endl;
        exit(1);
    }

    // Process each hidden neuron
    for (int i = 0; i < hidden_size; i++) {
        if (i >= hidden_neurons->size()) {
            std::cerr << "Error: Hidden neurons out of bounds at index " << i << std::endl;
            exit(1);
        }

        neuron* n = (*hidden_neurons)[i];
        double neuron_output = n->activate(input);  // Pass input to the neuron for activation

        // Calculating the gates for LSTM using the neuron output and hidden state
        (*forget_gate)[i] = sigmoid(neuron_output + (*hidden_state)[i]);  // Forget gate
        (*input_gate)[i] = sigmoid(neuron_output + (*hidden_state)[i]);   // Input gate
        (*output_gate)[i] = sigmoid(neuron_output + (*hidden_state)[i]);  // Output gate
        (*cell_candidate)[i] = std::tanh(neuron_output + (*hidden_state)[i]);  // Candidate memory

        // Updating cell state
        (*cell_state)[i] = (*forget_gate)[i] * (*cell_state)[i] + (*input_gate)[i] * (*cell_candidate)[i];

        // Updating hidden state (output of this time step)
        (*hidden_state)[i] = (*output_gate)[i] * std::tanh((*cell_state)[i]);

        // Set output for the current time step
        (*output)[i] = (*hidden_state)[i];
    }

    // Clean up
    delete forget_gate;
    delete input_gate;
    delete output_gate;
    delete cell_candidate;

    return output;
}

std::vector<double>* RNNLayer::lstm_backward(std::vector<double>* d_output, std::vector<double>* d_next_cell_state) {
    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);  // Gradients w.r.t input
    std::vector<double>* d_hidden_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t hidden state
    std::vector<double>* d_cell_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t cell state

    std::vector<double> d_forget_gate(hidden_size, 0.0);
    std::vector<double> d_input_gate(hidden_size, 0.0);
    std::vector<double> d_output_gate(hidden_size, 0.0);
    std::vector<double> d_cell_candidate(hidden_size, 0.0);

    // Loop through each neuron in the hidden layer to compute gradients
    for (int i = 0; i < hidden_size; ++i) {
        neuron* current_neuron = (*hidden_neurons)[i];  // Access the current hidden neuron
        if (!current_neuron) {
            std::cerr << "Error: Null neuron at index " << i << std::endl;
            exit(1);
        }

        // Get the gradient for the output gate and backpropagate through it
        d_output_gate[i] = (*d_output)[i] * std::tanh((*cell_state)[i]);  // Gradient of the output gate
        double d_tanh_cell_state = (*d_output)[i] * (*hidden_state)[i];  // Gradient of tanh(cell state)

        // Compute gradients for forget gate, input gate, and candidate cell state
        d_forget_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_input_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_cell_candidate[i] = d_tanh_cell_state * (1 - std::pow(std::tanh((*cell_state)[i]), 2));  // Derivative of tanh

        // Accumulate gradients for the next time step
        (*d_cell_state)[i] = (*d_next_cell_state)[i] * d_forget_gate[i];  // Accumulating cell state gradients

        // Update weights for neurons based on the calculated gradients
        for (size_t j = 0; j < current_neuron->weights->size(); ++j) {
            current_neuron->weights->at(j) -= learning_rate * d_forget_gate[i];  // Update weights for forget gate
            current_neuron->weights->at(j) -= learning_rate * d_input_gate[i];   // Update weights for input gate
            current_neuron->weights->at(j) -= learning_rate * d_output_gate[i];  // Update weights for output gate
        }

        // Propagate the gradient back to the input for this time step
        for (size_t k = 0; k < current_neuron->weights->size(); ++k) {
            (*d_input)[k] += current_neuron->weights->at(k) * d_output_gate[i];
        }
    }

    return d_input;  // Return the input gradients for the previous layer
}

double RNNLayer::sigmoid(double x){
    return 1.0 / (1.0 + std::exp(-x));
}