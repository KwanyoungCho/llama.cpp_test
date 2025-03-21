/* File: llama.cpp/tests/test_kv_cache.cpp */
#include <iostream>
#include "block_kv_cache.h"

int main() {
    // 설정: num_blocks, block_size, num_kv_heads, head_size
    size_t num_blocks = 2;
    size_t block_size = 4;      // 블록당 4개의 토큰 저장
    size_t num_kv_heads = 2;
    size_t head_size = 3;       // 각 head당 3개의 값
    
    // BlockKVCache 인스턴스 생성
    BlockKVCache cache(num_blocks, block_size, num_kv_heads, head_size);
    cache.initCache();

    // 각 블록 데이터 크기 계산
    const size_t dataSize = block_size * num_kv_heads * head_size; // 4 x 2 x 3 = 24

    // dummy data 생성: 첫 번째 블록
    float key[dataSize];
    float value[dataSize];
    for (size_t i = 0; i < dataSize; i++) {
        key[i] = static_cast<float>(i);
        value[i] = static_cast<float>(i) * 2.0f;
    }

    // insertKVBlock 테스트: 첫 번째 삽입
    VirtualBlockEntry entry1 = cache.insertKVBlock(key, value);
    std::cout << "Inserted block at physical index: " << entry1.physicalBlockIndex
              << ", slot: " << entry1.slotIndex << std::endl;

    // 두 번째 블록용 dummy data
    float key2[dataSize];
    float value2[dataSize];
    for (size_t i = 0; i < dataSize; i++) {
        key2[i] = static_cast<float>(i) * 3.0f;
        value2[i] = static_cast<float>(i) * 4.0f;
    }
    VirtualBlockEntry entry2 = cache.insertKVBlock(key2, value2);
    std::cout << "Inserted block at physical index: " << entry2.physicalBlockIndex
              << ", slot: " << entry2.slotIndex << std::endl;

    // 세 번째 삽입 - 만약 기존 슬롯이 부족하면 새로운 PhysicalBlock이 할당되어야 함
    VirtualBlockEntry entry3 = cache.insertKVBlock(key, value);
    std::cout << "Inserted block at physical index: " << entry3.physicalBlockIndex
              << ", slot: " << entry3.slotIndex << std::endl;

    // 두 번째 블록을 해제한 후 재삽입 테스트
    cache.freeKVBlock(entry2);
    std::cout << "Freed block at physical index: " << entry2.physicalBlockIndex
              << ", slot: " << entry2.slotIndex << std::endl;

    VirtualBlockEntry entry4 = cache.insertKVBlock(key2, value2);
    std::cout << "Inserted block at physical index: " << entry4.physicalBlockIndex
              << ", slot: " << entry4.slotIndex << std::endl;

    std::cout << "Test completed." << std::endl;
    return 0;
} 