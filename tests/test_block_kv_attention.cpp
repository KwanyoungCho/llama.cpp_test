/*
 * Test for blockKVAttentionFwd functionality
 * - Create a BlockKVCache instance with parameters: num_blocks=4, block_size=2, num_kv_heads=1, head_size=4
 * - Initialize dummy query, key, and value data
 * - Use a slot mapping to update the cache in blockKVAttentionFwd
 * - Dummy attention will copy query to output, and debug messages will be printed if DEBUG_KV_CACHE is defined
 *
 * Build command (from llama.cpp directory):
 *   g++ -I./src tests/test_block_kv_attention.cpp src/block_kv_attention.cpp src/block_kv_cache.cpp -o test_block_kv_attention -DDEBUG_KV_CACHE
 */


#include <iostream>
#include <vector>
#include "../src/block_kv_attention.h"
#include "../src/block_kv_cache.h"

int main() {
    // Parameters for BlockKVCache
    const size_t num_blocks = 4;
    const size_t block_size = 2;
    const size_t num_kv_heads = 1;
    const size_t head_size = 4;

    // Create BlockKVCache instance and initialize
    BlockKVCache cache(num_blocks, block_size, num_kv_heads, head_size);
    cache.initCache();

    // Dummy data for query, key, value
    // Let's assume query_size equals 8 elements (arbitrary)
    const size_t query_size = 8;
    std::vector<float> query(query_size);
    for (size_t i = 0; i < query_size; i++) {
        query[i] = static_cast<float>(i + 1);
    }

    // For key/value data, assume one block's worth of data (block_data_size)
    const size_t block_data_size = block_size * num_kv_heads * head_size;
    std::vector<float> keyData(block_data_size);
    std::vector<float> valueData(block_data_size);
    for (size_t i = 0; i < block_data_size; i++) {
        keyData[i] = static_cast<float>((i + 1) * 10);
        valueData[i] = static_cast<float>((i + 1) * 100);
    }

    // Slot mapping: we'll map the single block to slot 1 in the cache
    std::vector<int> slotMapping = {1};

    // Output buffer same size as query
    std::vector<float> output(query_size, 0.0f);

    // Call the dummy blockKVAttentionFwd function
    blockKVAttentionFwd(query.data(), query_size, cache,
                        keyData.data(), valueData.data(),
                        slotMapping, output.data());

    // Print the output; it should be identical to the query (dummy attention)
    std::cout << "Output of blockKVAttentionFwd:" << std::endl;
    for (size_t i = 0; i < output.size(); i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
} 