#include "../include/network.hpp"
#include <algorithm>

network::network(std::vector<int> *spec, int input_size, int num_classes, double learning_rate) {
    this->input_size = input_size;
    this->num_classes = num_classes;
    this->learning_rate = learning_rate;
    layers = new std::vector<layer *>();

    std::cout << "Starting layer initialization..." << std::endl;

    int current_input_size = input_size;  // Track the input size for each layer

    // Initialize the ConvLayer first
    int conv_filter_size = 3;  // Use filter size 3 for convolution
    int conv_num_filters = spec->at(0);  // First element in spec is the number of filters
    layers->push_back(new ConvLayer(conv_filter_size, conv_num_filters, learning_rate));

    std::cout << "ConvLayer initialized with filter size " << conv_filter_size 
              << " and " << conv_num_filters << " filters." << std::endl;

    // Now, get the output size after convolution and pooling (or flattening if pooling is removed)
    ConvLayer* conv_layer = static_cast<ConvLayer*>(layers->back());

    // Run a dummy forward pass to get the size after convolution and pooling/flattening
    std::vector<double> dummy_input(input_size, 0.0);  // Create a dummy input with the original input size
    std::vector<double>* conv_output = conv_layer->forward(&dummy_input);
    int pooled_output_size = conv_output->size();  // Get the size after convolution and pooling/flattening

    std::cout << "Pooled output size from ConvLayer: " << pooled_output_size << std::endl;

    // Set up chunking for RNN input size if necessary
    int chunk_size = 8192;  // Adjust chunk size based on available resources
    int num_chunks = pooled_output_size / chunk_size;
    int remaining_elements = pooled_output_size % chunk_size;

    std::cout << "Number of chunks: " << num_chunks << ", Remaining elements: " << remaining_elements << std::endl;

    // Use the pooled output size as the input size for the first RNNLayer
    current_input_size = chunk_size;

    // Initialize RNN layers in chunks
    for (int i = 1; i < spec->size(); i++) {
        int hidden_size = spec->at(i);  // Hidden size for each layer
        std::cout << "Initializing RNNLayer with input size (chunked) " << current_input_size 
                  << " and hidden size " << hidden_size << "." << std::endl;

        layers->push_back(new RNNLayer(current_input_size, hidden_size, learning_rate));

        current_input_size = hidden_size;  // Update input size for the next RNNLayer
    }

    // Initialize the final output layer (RNNLayer) in chunks
    std::cout << "Initializing final RNNLayer with input size (chunked) " << current_input_size 
              << " and output size " << num_classes << "." << std::endl;
    layers->push_back(new RNNLayer(current_input_size, num_classes, learning_rate));

    std::cout << "Network initialized with " << layers->size() << " layers." << std::endl;

    delete conv_output;  // Clean up the dummy input/output after determining sizes
}

network::~network() {
    for (layer *l : *layers) {
        delete l;
    }
    delete layers;
    close_debug_output();
}

std::vector<double>* network::fprop(data *d) {
    // Create the input vector from the feature vector
    std::vector<double>* input = new std::vector<double>(d->get_feature_vector()->begin(), d->get_feature_vector()->end());
    std::cout << "Forward Pass | Initial Feature Vector Size: " << input->size() << std::endl;

    // Iterate over each layer in the network
    for (int i = 0; i < layers->size(); ++i) {
        layer* current_layer = (*layers)[i];

        if (RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(current_layer)) {
            std::cout << "Processing RNNLayer with dynamic chunking." << std::endl;

            // Dynamically calculate chunks without relying on `get_num_chunks`
            int max_chunks = 50;  // Upper limit for number of chunks
            int chunk_size = std::max(1, static_cast<int>(input->size() / max_chunks));  // Ensure a valid chunk size
            int calculated_chunks = (input->size() + chunk_size - 1) / chunk_size;  // Round up to account for remaining elements

            // Set the calculated chunks
            std::cout << "Calculated chunks: " << calculated_chunks << std::endl;
            cd->set_num_chunks(&calculated_chunks);  // Use the calculated chunks directly
            std::cout << "Number of chunks set to: " << cd->get_num_chunks() << std::endl;

            // Process input data in chunks
            std::vector<double>* rnn_output = new std::vector<double>();
            for (int chunk_index = 0; chunk_index < cd->get_num_chunks(); ++chunk_index) {
                int chunk_start = chunk_index * chunk_size;
                int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(input->size()));

                std::cout << "Processing chunk " << chunk_index + 1 << "/" << cd->get_num_chunks()
                          << " with start: " << chunk_start << " and end: " << chunk_end << std::endl;

                // Extract the chunk and pass it to the RNN layer
                std::vector<double> chunk(input->begin() + chunk_start, input->begin() + chunk_end);
                std::vector<double>* chunk_output = rnn_layer->forward(&chunk);

                // Append the chunk output to the overall RNN output
                rnn_output->insert(rnn_output->end(), chunk_output->begin(), chunk_output->end());
                delete chunk_output;
            }

            // Replace input with the RNN output for the next layer
            delete input;
            input = rnn_output;

        } else {
            // Forward pass for non-RNN layers
            input = current_layer->forward(input);
        }

        // Check if input becomes invalid or empty
        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector became empty after forward pass in layer " << i << "!" << std::endl;
            throw std::runtime_error("Forward pass failed due to empty input");
        }

        std::cout << "Layer " << i << " forward pass completed with output size: " << input->size() << std::endl;
    }

    return input;
}

void network::initialize_chunks(common_data* cd, std::vector<double>* input) {
    int max_chunks = 50;  // Upper limit for number of chunks
    int chunk_size = std::max(1, static_cast<int>(input->size() / max_chunks));  // Ensure a valid chunk size
    int calculated_chunks = (input->size() + chunk_size - 1) / chunk_size;      // Round up for remaining elements

    cd->set_num_chunks(&calculated_chunks);
    std::cout << "Chunks initialized: " << cd->get_num_chunks() << " with chunk size: " << chunk_size << std::endl;
}

void network::bprop(data *d) {
    int num_chunks = get_num_chunks();  // Get num_chunks from common_data
    std::vector<double>* output = fprop(d);  // Forward pass
    std::cout << "bprop" << std::endl;

    std::vector<double>* gradients = new std::vector<double>(output->size(), 0.0);  // Initialize gradients
    std::cout << "gradient" << std::endl;
    std::vector<int>* class_vector = new std::vector<int>();
    std::cout << "class vector" << std::endl;

    // First, check if the vector pointer is valid and not null
    if (d->get_class_vector() == nullptr) {
        std::cerr << "Error: class_vector is nullptr" << std::endl;
    } else {
        std::cout << "else" << std::endl;
        // Now that we know it's valid, we can safely access the size and loop over it
        for (int i = 0; i < d->get_class_vector()->size(); i++) {
            std::cout << "class_vector_size: " << d->get_class_vector()->size() << std::endl;
            std::cout << "Class vector at i: " << ((*d->get_class_vector())[i]) << std::endl;
            class_vector->push_back((*d->get_class_vector())[i]);  // Dereference the vector to get the data
        }
    }

    // Calculate the gradients of the output layer based on the loss
    if (output == nullptr) {
    std::cerr << "Error: Output is null!" << std::endl;
    return;  // Exit or handle the error accordingly
    }

    if (class_vector == nullptr) {
        std::cerr << "Error: Class vector is null!" << std::endl;
        return;  // Exit or handle the error accordingly
    }

    for (size_t i = 0; i < output->size(); i++) {
        double error = (*output)[i] - (*class_vector)[i];  // Error calculation (output - label)
        (*gradients)[i] = error * transfer_derivative((*output)[i]);  // Apply the transfer derivative
    }

    // Now backpropagate the gradients through each layer
    for (int i = layers->size() - 1; i >= 0; i--) {
        layer* current_layer = (*layers)[i];

        // Apply the gradients to the current layer
        gradients = current_layer->backward(gradients);  // The backward method of each layer should update weights
    }

    // Here, you don't need to delete gradients anymore, as they are being handled by each layer's backward function.
    // If needed, you can delete the gradients after they are no longer needed.
}

void network::update_weights(data *d) {
    for(int i = 0; i < layers->size(); i++){
        layer* current_layer = (*layers)[i];

        //Check if current layer is a ConvLayer or RNNLayer
        if(ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(current_layer)){
            //This is a Convolutional Layer, update weights for ConvLayer
            for(int f = 0; f < conv_layer->num_filters; f++){
                for(int j = 0; j < conv_layer->filter_size * conv_layer->filter_size; j++){
                    for(auto &n : (*conv_layer->filters)[f]){
                        //This is the rule for the simple weight update
                        n->weights->at(j) -= learning_rate * n->delta;
                    }
                }
            }
        }
        else if(RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(current_layer)){
            //Update weights for RNNLayer
            for(int h = 0; h < rnn_layer->hidden_size; h++){
                for(int w = 0; w < rnn_layer->input_size; w++){
                    (*rnn_layer->hidden_neurons)[h]->weights->at(w) -= learning_rate * (*rnn_layer->hidden_neurons)[h]->delta;
                }
            }
        }
    }
}

double network::transfer(double activation) {
    return 1.0 / (1.0 + std::exp(-activation));
}

double network::transfer_derivative(double output) {
    return output * (1 - output);
}

int network::predict(data *d) {
    std::vector<double>* output = fprop(d);
    std::cout << "predict" << std::endl;
    return std::distance(output->begin(), std::max_element(output->begin(), output->end()));
}

// Training: Pass data through the network, forward pass only for now
void network::train(int epochs, double validation_threshold) {
    if (common_training_data == nullptr || common_training_data->empty()) {
        std::cerr << "Error: Training data is empty!" << std::endl;
        return;
    }

    int total_validation_samples = common_validation_data->size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;

        double total_loss = 0.0;
        int sample_index = 0;
        int total_samples = common_training_data->size();

        display_progress_bar(sample_index, total_samples);

        for (data* d : *common_training_data) {
            std::cout << "Processing sample " << sample_index + 1 << " of " << total_samples << std::endl;

            std::vector<double>* input = d->get_feature_vector();

            if (input == nullptr || input->empty()) {
                std::cerr << "Error: Input feature vector is empty for sample " << sample_index << std::endl;
                continue;
            }

            int max_chunks = 50;
            int chunk_size = std::max(1, static_cast<int>(input->size() / max_chunks));
            int calculated_chunks = (input->size() + chunk_size - 1) / chunk_size;

            std::cout << "Setting num_chunks to: " << calculated_chunks << std::endl;
            cd->set_num_chunks(&calculated_chunks);
            std::cout << "Number of chunks set to: " << cd->get_num_chunks() << std::endl;

            std::vector<double>* output = nullptr;
            try {
                output = fprop(d);
            } catch (const std::exception& e) {
                std::cerr << "An exception occurred during fprop: " << e.what() << std::endl;
                delete output;
                continue;
            }

            if (output == nullptr || output->empty()) {
                std::cerr << "Error: Output feature vector is empty or null for sample " << sample_index << std::endl;
                delete output;
                continue;
            }

            double sample_loss = calculate_loss(output, d->get_class_vector());
            total_loss += sample_loss;
            std::cout << "Sample loss: " << sample_loss << std::endl;

            try {
                bprop(d);
                std::cout << "Backward pass completed for sample " << sample_index << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "An exception occurred during bprop: " << e.what() << std::endl;
            }

            update_weights(d);
            std::cout << "Weights updated for sample " << sample_index << std::endl;

            sample_index++;
            display_progress_bar(sample_index, total_samples);

            delete output;
        }

        std::cout << "\nEpoch " << epoch + 1 << " completed. Total loss: " << total_loss << std::endl;

        // Validation phase
        double correct_predictions = 0.0;
        for (data* vd : *common_validation_data) {
            correct_predictions += validate(vd);
        }

        double validation_accuracy = correct_predictions / total_validation_samples;
        std::cout << "Validation Accuracy after epoch " << epoch + 1 << ": " << validation_accuracy * 100 << "%" << std::endl;

        if (validation_accuracy >= validation_threshold) {
            std::cout << "Stopping early as validation accuracy reached " << validation_accuracy * 100 << "%" << std::endl;
            break;
        }
    }
}


void network::set_debug_output(const std::string &filename) {
    debug_output.open(filename);
    if (!debug_output.is_open()) {
        std::cerr << "Error opening debug file: " << filename << std::endl;
    }
}

void network::close_debug_output() {
    if (debug_output.is_open()) {
        debug_output.close();
    }
}

void network::save_model(const std::string &filename) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()){
        std::cerr << "Error: Could not open file " << filename << " for saving model." << std::endl;
        exit(1);
    }

    int total_layers = layers->size();
    int processed_layers = 0;

    // Save each layer's weights
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);

        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double weight : *n->weights){
                        outfile << weight << " ";
                    }
                    outfile << std::endl;
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double weight : *n->weights){
                    outfile << weight << " ";
                }
                outfile << std::endl;
            }
        }

        processed_layers++;
        display_progress_bar(processed_layers, total_layers);  // Update progress bar for saving weights
    }

    outfile.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void network::load_model(const std::string &filename) {
    std::ifstream infile(filename);
    if(!infile.is_open()){
        std::cerr << "Error: could not open file " << filename << " for loading model." << std::endl;
        exit(1);
    }
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);

        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double& weight : *n->weights){
                        infile >> weight;
                    }
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double& weight : *n->weights){
                    infile >> weight;
                }
            }
        }
    }
    infile.close();
    std::cout << "Model loaded from " << filename << std::endl;
}

double network::validate(data* d) {
    // Forward pass to get the predicted output
    std::vector<double>* output = nullptr;
    try {
        output = fprop(d);  // Forward pass through the network
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during fprop: " << e.what() << std::endl;
        return 0.0;  // Return 0.0 if forward pass fails
    }

    // Check if the output is valid
    if (output == nullptr || output->empty()) {
        std::cerr << "Error: Output feature vector is empty or null!" << std::endl;
        return 0.0;  // Return 0.0 if output is invalid
    }

    // Find the index of the maximum value in the output vector (predicted class)
    auto max_element_it = std::max_element(output->begin(), output->end());
    int predicted_class_index = std::distance(output->begin(), max_element_it);

    // Print for debugging purposes
    std::cout << "Predicted class index: " << predicted_class_index << std::endl;
    std::cout << "Actual class: " << d->get_class_vector()->at(0) << std::endl;

    // Compare predicted class with the actual class
    if (predicted_class_index == d->get_class_vector()->at(0)) {
        std::cout << "Prediction correct!" << std::endl;
        return 1.0;  // Accuracy for this sample (correct prediction)
    } else {
        std::cout << "Prediction incorrect!" << std::endl;
        return 0.0;  // Accuracy for this sample (incorrect prediction)
    }
}

double network::test() {
    if (common_testing_data == nullptr || common_testing_data->empty()) {
        std::cerr << "Error: Testing data is empty!" << std::endl;
        return 0.0;
    }

    int correct = 0;
    int total = 0;

    // Iterate over the testing data in common_testing_data
    for (data* d : *common_testing_data) {
        std::vector<double>* input = d->get_feature_vector();
        
        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector is empty!" << std::endl;
            continue;
        }

        // Forward pass through the network
        std::vector<double>* output = fprop(d);
        std::cout << "test" << std::endl;

        // Get the predicted class
        int predicted_class = std::distance(output->begin(), std::max_element(output->begin(), output->end()));

        if (d->get_class_vector()->at(predicted_class) == 1) {
            correct++;
        }
        total++;
    }

    return static_cast<double>(correct) / total;
}

void network::output_predictions(const std::string &filename, data_handler *dh) {
    // Output predictions to CSV or other file formats
}

double network::calculate_loss(std::vector<double>* output, std::vector<int>* class_vector) {
    double loss = 0.0;
    for (size_t i = 0; i < class_vector->size(); i++) {
        // Assuming binary cross-entropy loss for this example
        int target = (*class_vector)[i];  // Dereference to access the value
        double prediction = (*output)[i];
        
        // Binary cross-entropy loss calculation
        loss += -target * std::log(prediction) - (1 - target) * std::log(1 - prediction);
    }
    return loss;
}

// In Main
int main() {
    try {
        // Initialize the data handler
        data_handler *dh = new data_handler();
        std::cout << "Starting to read the labels" << std::endl;
        dh->read_data_and_labels("E:\\Code\\VOOD\\data\\binary\\output_binary_file.data", "E:\\Code\\VOOD\\data\\labels\\VOOD_labels.csv");
        dh->split_data();

        // Initialize the network
        std::vector<int> *spec = new std::vector<int>{128, 64, 3};
        network *net = new network(spec, dh->get_training_data()->at(0)->get_feature_vector()->size(), 3, 0.01);

        // Set debug output file
        net->set_debug_output("debug_output.txt");

        // Assign the split data to the network's inherited common_data members
        net->set_common_training_data(dh->get_training_data());
        net->set_common_testing_data(dh->get_testing_data());
        net->set_common_validation_data(dh->get_validation_data());

        // Train the network
        std::cout << "Starting training..." << std::endl;
        net->train(10, 0.98);  // Train for 10 epochs

        // Save the trained model
        net->save_model("E:\\Code\\VOOD\\data\\saved_model\\trained_model.bin");
        std::cout << "Model saved successfully." << std::endl;

        // Test the model on test data
        double test_accuracy = net->test();
        std::cout << "Test Accuracy: " << test_accuracy << std::endl;

        // Cleanup
        delete dh;
        delete net;
        delete spec;
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred!" << std::endl;
    }

    return 0;
}
