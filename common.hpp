#ifndef __COMMON_HPP
#define __COMMON_HPP

#include "data.hpp"
#include <vector>

void display_progress_bar(long long processed, long long total);

class common_data{
    protected:
        std::vector<data *> *common_training_data;
        std::vector<data *> *common_testing_data;
        std::vector<data *> *common_validation_data;
        
        int num_chunks;  // Add num_chunks to manage chunking across the whole data

    public:
        common_data();
        ~common_data();
        
        std::vector<data *> *get_common_training_data();
        std::vector<data *> *get_common_testing_data();
        std::vector<data *> *get_common_validation_data();

        // Setters
        void set_common_training_data(std::vector<data *> *vect);
        void set_common_testing_data(std::vector<data *> *vect);
        void set_common_validation_data(std::vector<data *> *vect);

        // Getter and Setter for num_chunks
        int get_num_chunks() const;
        void set_num_chunks(int* chunk_count_ptr);
};

#endif
