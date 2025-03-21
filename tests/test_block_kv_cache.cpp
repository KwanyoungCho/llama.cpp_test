/*
 * Test for BlockKVCache functionality
 * - Create two BlockKVCache objects with parameters:
 *     num_blocks = 4, block_size = 2, num_kv_heads = 1, head_size = 4
 * - Write dummy key and value data to specific slots using writeToCache
 * - Print the cache contents
 * - Perform swapBlocks between the two caches
 * - Perform copyBlocks operation and print the updates
 *
 * Compile with DEBUG_KV_CACHE defined for debugging output.
 */



#include <iostream>
#include <vector>
#include "../src/block_kv_cache.h"

int main() {
    // Parameters for BlockKVCache
    const size_t num_blocks = 4;
    const size_t block_size = 2;
    const size_t num_kv_heads = 1;
    const size_t head_size = 4;

    // Create two cache objects
    BlockKVCache cache1(num_blocks, block_size, num_kv_heads, head_size);
    BlockKVCache cache2(num_blocks, block_size, num_kv_heads, head_size);

    cache1.initCache();
    cache2.initCache();

    const size_t block_data_size = block_size * num_kv_heads * head_size;

    // Prepare dummy data for writeToCache for cache1
    // We'll write 2 blocks of data
    const size_t n_test_blocks = 2;
    std::vector<float> test_keys(n_test_blocks * block_data_size);
    std::vector<float> test_values(n_test_blocks * block_data_size);
    // Fill dummy data: for block i, values = (i+1)*10 + j for keys, (i+1)*100 + j for values
    for (size_t i = 0; i < n_test_blocks; i++) {
        for (size_t j = 0; j < block_data_size; j++) {
            test_keys[i * block_data_size + j] = static_cast<float>((i + 1) * 10 + j);
            test_values[i * block_data_size + j] = static_cast<float>((i + 1) * 100 + j);
        }
    }

    // Use slot mapping: assign test blocks to slots 1 and 3 in cache1
    std::vector<int> slot_mapping = {1, 3};
    cache1.writeToCache(test_keys.data(), test_values.data(), slot_mapping);

    // Print cache1 key and value caches
    std::cout << "Cache1 key_cache after writeToCache:" << std::endl;
    const auto &key_cache1 = cache1.getKeyCache();
    for (size_t i = 0; i < key_cache1.size(); i++) {
        std::cout << key_cache1[i] << " ";
        if ((i + 1) % block_data_size == 0)
            std::cout << std::endl;
    }
    std::cout << "Cache1 value_cache after writeToCache:" << std::endl;
    const auto &value_cache1 = cache1.getValueCache();
    for (size_t i = 0; i < value_cache1.size(); i++) {
        std::cout << value_cache1[i] << " ";
        if ((i + 1) % block_data_size == 0)
            std::cout << std::endl;
    }

    // Prepare dummy data for cache2
    // Write 2 blocks of data with different values
    std::vector<float> test_keys2(n_test_blocks * block_data_size);
    std::vector<float> test_values2(n_test_blocks * block_data_size);
    for (size_t i = 0; i < n_test_blocks; i++) {
        for (size_t j = 0; j < block_data_size; j++) {
            test_keys2[i * block_data_size + j] = static_cast<float>((i + 1) * 20 + j);
            test_values2[i * block_data_size + j] = static_cast<float>((i + 1) * 200 + j);
        }
    }
    // Use slot mapping: assign to slots 0 and 2 in cache2
    std::vector<int> slot_mapping2 = {0, 2};
    cache2.writeToCache(test_keys2.data(), test_values2.data(), slot_mapping2);

    std::cout << "Cache2 key_cache after writeToCache:" << std::endl;
    const auto &key_cache2 = cache2.getKeyCache();
    for (size_t i = 0; i < key_cache2.size(); i++) {
        std::cout << key_cache2[i] << " ";
        if ((i + 1) % block_data_size == 0)
            std::cout << std::endl;
    }
    std::cout << "Cache2 value_cache after writeToCache:" << std::endl;
    const auto &value_cache2 = cache2.getValueCache();
    for (size_t i = 0; i < value_cache2.size(); i++) {
        std::cout << value_cache2[i] << " ";
        if ((i + 1) % block_data_size == 0)
            std::cout << std::endl;
    }

    // Test swapBlocks: swap blocks between cache1 and cache2
    // For simplicity, swap mapping: swap cache1 block indices 1 and 3 with cache2 block indices 0 and 2 respectively
    std::vector<int> swap_mapping = {0, 2};
    cache1.swapBlocks(cache2, swap_mapping);

    std::cout << "After swapBlocks:" << std::endl;
    std::cout << "Cache1 key_cache:" << std::endl;
    for (size_t i = 0; i < key_cache1.size(); i++) {
        std::cout << key_cache1[i] << " ";
        if ((i + 1) % block_data_size == 0)
            std::cout << std::endl;
    }
    std::cout << "Cache2 key_cache:" << std::endl;
    for (size_t i = 0; i < key_cache2.size(); i++) {
        std::cout << key_cache2[i] << " ";
        if ((i + 1) % block_data_size == 0)
            std::cout << std::endl;
    }

    // Test copyBlocks: copy block index 2 from cache2 to cache1
    std::vector<int> copy_mapping = {2};
    cache1.copyBlocks(cache2, copy_mapping);

    std::cout << "After copyBlocks (copying block index 2 from cache2 to cache1):" << std::endl;
    std::cout << "Cache1 key_cache:" << std::endl;
    for (size_t i = 0; i < key_cache1.size(); i++) {
        std::cout << key_cache1[i] << " ";
        if ((i + 1) % block_data_size == 0)
            std::cout << std::endl;
    }

    return 0;
} 