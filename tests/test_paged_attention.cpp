/* File: llama.cpp/tests/test_paged_attention.cpp */
#include <iostream>
#include "block_kv_attention.h"

int main() {
    // 설정: num_blocks, block_size, num_kv_heads, head_size
    size_t num_blocks = 2;
    size_t block_size = 4;      // 블록당 4개의 토큰 저장
    size_t num_kv_heads = 2;
    size_t head_size = 3;       // 각 head당 3개의 값

    // BlockKVCache 인스턴스 생성 및 초기화
    BlockKVCache cache(num_blocks, block_size, num_kv_heads, head_size);
    cache.initCache();

    // query 데이터: 전체 query 크기는 25
    size_t query_size = 25;
    float query[25];
    for (size_t i = 0; i < query_size; i++) {
        query[i] = static_cast<float>(i);
    }

    // key와 value 데이터 준비 (각 블록 데이터 크기 계산)
    const size_t dataSize = block_size * num_kv_heads * head_size; // 4*2*3 = 24
    float key[dataSize];
    float value[dataSize];
    for (size_t i = 0; i < dataSize; i++) {
        key[i] = static_cast<float>(i + 1);
        value[i] = static_cast<float>((i + 1) * 2);
    }

    // output 버퍼 초기화
    float output[25] = {0};

    // page_size를 10으로 설정 (한 페이지 당 10개의 query 요소 처리)
    size_t page_size = 10;

    // paged attention 함수 호출
    blockKVAttentionPagedFwd(query, query_size, cache, key, value, output, page_size);

    // 출력 결과 확인 (현재 dummy attention이므로 query가 그대로 복사됨)
    std::cout << "Paged attention output:" << std::endl;
    for (size_t i = 0; i < query_size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
} 