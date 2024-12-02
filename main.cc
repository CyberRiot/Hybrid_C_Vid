#include <iostream>
#include <vector>
#include <string>
#include "../include/network.hpp"
#include "../include/data_handler.hpp"

void run_model_against_data(network& trained_model, const std::string& test_data_path, const std::string& label_file_path) {
    // Load test data and labels
    data_handler handler;
    handler.read_data_and_labels(test_data_path, label_file_path);

    std::vector<data*>* test_data = handler.get_testing_data();
    if (test_data->empty()) {
        std::cerr << "No test data found. Ensure the test data and label files are correctly loaded." << std::endl;
        return;
    }

    std::cout << "Running model inference on test data..." << std::endl;
    for (size_t i = 0; i < test_data->size(); ++i) {
        data* sample = (*test_data)[i];
        
        try {
            // Perform a forward pass on the test sample
            std::vector<double>* output = trained_model.fprop(sample);
            
            // Display results
            std::cout << "Sample " << i + 1 << "/" << test_data->size() << " | Output: ";
            for (double value : *output) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
            
            delete output; // Clean up the output vector
        } catch (const std::exception& e) {
            std::cerr << "Error during inference on sample " << i + 1 << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Inference complete." << std::endl;
}

// Example usage
int main() {
    network trained_model; // Assume the network is already trained and initialized
    std::string test_data_path = "test_data.bin";  // Replace with your test data file path
    std::string label_file_path = "test_labels.csv"; // Replace with your label file path

    run_model_against_data(trained_model, test_data_path, label_file_path);
    return 0;
}
