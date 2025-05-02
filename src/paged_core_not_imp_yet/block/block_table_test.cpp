#include "block_table.h"
#include "naive_block.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

// Block::Factory 구현
class NaiveBlockFactory : public Block::Factory {
public:
    std::shared_ptr<Block> operator()(
        std::shared_ptr<Block> prev_block,
        const std::vector<int>& token_ids,
        int block_size,
        BlockAllocator* allocator,
        int block_id,
        bool is_immutable = false) override {
        return std::make_shared<NaiveBlock>(
            prev_block, token_ids, block_size, allocator, block_id, 0);
    }
};

void test_block_table_basic_operations() {
    // 기본 설정
    const int block_size = 4;
    const int num_blocks = 10;
    auto allocator = std::make_shared<NaiveBlockAllocator>(
        std::make_shared<NaiveBlockFactory>(),
        num_blocks,
        block_size,
        std::vector<int>(),
        nullptr
    );

    // BlockTable 생성
    BlockTable table(block_size, static_cast<DeviceAwareBlockAllocator*>(allocator.get()), std::vector<std::shared_ptr<Block>>(), -1);

    // 테스트 1: 기본 할당
    std::vector<int> token_ids = {1, 2, 3, 4, 5, 6, 7, 8};
    table.allocate(token_ids, Device::CPU);
    
    assert(table.num_full_slots() == token_ids.size());
    assert(table.physical_block_ids().size() == 2);  // 8개의 토큰은 2개의 블록 필요

    // 테스트 2: 토큰 추가
    std::vector<int> additional_tokens = {9, 10, 11, 12};
    table.append_token_ids(additional_tokens, 0, 0);
    
    assert(table.num_full_slots() == token_ids.size() + additional_tokens.size());
    assert(table.physical_block_ids().size() == 3);  // 12개의 토큰은 3개의 블록 필요

    std::cout << "기본 작업 테스트 통과!\n";
}

void test_block_table_advanced_features() {
    // 기본 설정
    const int block_size = 4;
    const int num_blocks = 10;
    auto allocator = std::make_shared<NaiveBlockAllocator>(
        std::make_shared<NaiveBlockFactory>(),
        num_blocks,
        block_size,
        std::vector<int>(),
        nullptr
    );

    // BlockTable 생성
    BlockTable table(block_size, static_cast<DeviceAwareBlockAllocator*>(allocator.get()), std::vector<std::shared_ptr<Block>>(), 2);  // max_block_sliding_window = 2

    // 테스트 1: fork 기능
    std::vector<int> token_ids = {1, 2, 3, 4, 5, 6, 7, 8};
    table.allocate(token_ids, Device::CPU);
    
    auto forked_table = table.fork();
    assert(forked_table->num_full_slots() == table.num_full_slots());
    assert(forked_table->physical_block_ids().size() == table.physical_block_ids().size());

    // 테스트 2: sliding window
    std::vector<int> additional_tokens = {9, 10, 11, 12, 13, 14, 15, 16};
    table.append_token_ids(additional_tokens, 0, 8);  // 8개의 토큰이 계산됨
    
    assert(table.num_full_slots() == token_ids.size() + additional_tokens.size());
    // sliding window가 적용되어 첫 번째 블록이 해제되어야 함
    assert(table.physical_block_ids().size() <= 4);  // 16개의 토큰은 최대 4개의 블록 필요

    std::cout << "고급 기능 테스트 통과!\n";
}

void test_block_table_error_cases() {
    // 기본 설정
    const int block_size = 4;
    const int num_blocks = 10;
    auto allocator = std::make_shared<NaiveBlockAllocator>(
        std::make_shared<NaiveBlockFactory>(),
        num_blocks,
        block_size,
        std::vector<int>(),
        nullptr
    );

    // BlockTable 생성
    BlockTable table(block_size, static_cast<DeviceAwareBlockAllocator*>(allocator.get()), std::vector<std::shared_ptr<Block>>(), -1);

    // 테스트 1: 빈 토큰 리스트로 할당 시도
    try {
        table.allocate(std::vector<int>(), Device::CPU);
        assert(false);  // 예외가 발생해야 함
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()) == "token_ids should not be empty");
    }

    // 테스트 2: 할당 전에 토큰 추가 시도
    try {
        table.append_token_ids({1, 2, 3}, 0, 0);
        assert(false);  // 예외가 발생해야 함
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()) == "no blocks have been allocated");
    }

    std::cout << "에러 케이스 테스트 통과!\n";
}

int main() {
    try {
        test_block_table_basic_operations();
        test_block_table_advanced_features();
        test_block_table_error_cases();
        std::cout << "모든 테스트 통과!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "테스트 실패: " << e.what() << std::endl;
        return 1;
    }
} 