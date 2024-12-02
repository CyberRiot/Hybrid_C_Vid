#include "../include/common.hpp"

void display_progress_bar(long long processed, long long total) {
    if (total <= 0) {
        return;  // Prevent division by zero or negative values
    }

    int bar_width = 50;
    float progress = static_cast<float>(processed) / total;
    std::cout << "[";
    int pos = static_cast<int>(bar_width * progress);
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

common_data::common_data() {
    common_training_data = nullptr;
    common_testing_data = nullptr;
    common_validation_data = nullptr;
    num_chunks = 0;  // Initialize num_chunks
}

common_data::~common_data() {
    delete common_training_data;
    delete common_testing_data;
    delete common_validation_data;
}

std::vector<data *> *common_data::get_common_training_data() {
    return common_training_data;
}

std::vector<data *> *common_data::get_common_testing_data() {
    return common_testing_data;
}

std::vector<data *> *common_data::get_common_validation_data() {
    return common_validation_data;
}

void common_data::set_common_training_data(std::vector<data *> *vect) {
    common_training_data = vect;
}

void common_data::set_common_testing_data(std::vector<data *> *vect) {
    common_testing_data = vect;
}

void common_data::set_common_validation_data(std::vector<data *> *vect) {
    common_validation_data = vect;
}

int common_data::get_num_chunks() const {
    if (num_chunks <= 0) {
        std::cerr << "Warning: num_chunks is currently invalid or uninitialized! Current value: " << num_chunks << std::endl;
        throw std::runtime_error("Invalid or uninitialized num_chunks accessed in get_num_chunks.");
    }
    return num_chunks;
}

void common_data::set_num_chunks(int* chunk_count_ptr) {
    if (chunk_count_ptr == nullptr) {
        std::cerr << "Error: Null pointer passed to set_num_chunks." << std::endl;
        return;
    }

    if (*chunk_count_ptr < 0) {
        std::cerr << "Error: Attempted to set a negative value for num_chunks: " << *chunk_count_ptr << std::endl;
        return;
    }

    num_chunks = *chunk_count_ptr;
    std::cout << "num_chunks successfully set to: " << num_chunks << std::endl;
}
