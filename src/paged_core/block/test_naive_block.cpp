#include "naive_block.h"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

// DummyBlock: Block의 최소 구현 (토큰 저장 및 기본 getter, setter)
class DummyBlock : public Block {
public:
    DummyBlock(std::shared_ptr<Block> prev, int id, int capacity)
        : prev_block_(prev), block_id_(id), capacity_(capacity) {}

    void appendTokenIds(const std::vector<int>& token_ids) override {
        token_ids_.insert(token_ids_.end(), token_ids.begin(), token_ids.end());
    }

    int getBlockId() const override {
        return block_id_;
    }

    void setBlockId(int id) override {
        block_id_ = id;
    }

    std::vector<int>& getTokenIds() override {
        return token_ids_;
    }

    int numEmptySlots() const override {
        return capacity_ - static_cast<int>(token_ids_.size());
    }

    bool isFull() const override {
        return numEmptySlots() == 0;
    }

    std::shared_ptr<Block> getPrevBlock() const override {
        return prev_block_;
    }

    int numTokensTotal() const override {
        return token_ids_.size();
    }

    bool isComputed() const override {
        throw std::runtime_error("Not implemented: isComputed");
    }

    void setComputed(bool value) override {
        throw std::runtime_error("Not implemented: setComputed");
    }

    double getLastAccessed() const override {
        throw std::runtime_error("Not implemented: getLastAccessed");
    }

    void setLastAccessed(double timestamp) override {
        throw std::runtime_error("Not implemented: setLastAccessed");
    }

    int getContentHash() const override {
        return 0;
    }

private:
    std::vector<int> token_ids_;
    int block_id_;
    int capacity_;
    std::shared_ptr<Block> prev_block_;
};

// DummyBlockFactory: Block::Factory의 더미 구현으로, DummyBlock을 생성합니다.
class DummyBlockFactory : public Block::Factory {
public:
    std::shared_ptr<Block> operator()(std::shared_ptr<Block> prev_block,
                                      const std::vector<int>& token_ids,
                                      int block_size,
                                      BlockAllocator* allocator,
                                      int block_id = -1,
                                      bool computed = false) override {
        std::shared_ptr<DummyBlock> blk = std::make_shared<DummyBlock>(prev_block, block_id, block_size);
        blk->appendTokenIds(token_ids);
        return blk;
    }
};

int main() {
    // DummyBlockFactory를 생성 후 shared_ptr로 관리합니다.
    std::shared_ptr<Block::Factory> factory = std::make_shared<DummyBlockFactory>();

    // NaiveBlockAllocator 생성: 총 5개 블록, 각 블록 크기 100으로 초기화함.
    NaiveBlockAllocator allocator(factory, 5, 100);

    // 초기 상태 출력
    std::cout << "[초기 상태]" << std::endl;
    std::cout << "총 블록 수: " << allocator.get_num_total_blocks() << std::endl;
    std::cout << "남은 블록 수: " << allocator.get_num_free_blocks() << std::endl;
    std::cout << "블록 크기: " << allocator.blockSize() << std::endl;

    // allocate_mutable_block 테스트
    std::shared_ptr<Block> block1 = allocator.allocate_mutable_block(nullptr);
    std::cout << "\n[allocate_mutable_block 테스트]" << std::endl;
    std::cout << "블록1 ID: " << block1->getBlockId() << std::endl;

    // block1에 토큰 1~5 추가
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    block1->appendTokenIds(tokens);
    std::cout << "블록1 토큰 수: " << block1->getTokenIds().size() << std::endl;
    std::cout << "블록1 남은 슬롯: " << block1->numEmptySlots() << std::endl;
    std::cout << "블록1 전체 토큰 수: " << block1->numTokensTotal() << std::endl;
    std::cout << "블록1 isFull: " << (block1->isFull() ? "true" : "false") << std::endl;

    // setBlockId 테스트
    block1->setBlockId(999);
    std::cout << "블록1 ID 변경 후: " << block1->getBlockId() << std::endl;

    // allocate_immutable_block 테스트 (block1을 이전 블록으로 사용하여 토큰 6,7,8 추가)
    std::shared_ptr<Block> block2 = allocator.allocate_immutable_block(block1, {6, 7, 8});
    std::cout << "\n[allocate_immutable_block 테스트]" << std::endl;
    std::cout << "블록2 ID: " << block2->getBlockId() << std::endl;
    std::cout << "블록2 토큰 수: " << block2->getTokenIds().size() << std::endl;

    // 예외 테스트: isComputed와 getLastAccessed 호출
    try {
        block2->isComputed();
    } catch (const std::runtime_error& e) {
        std::cout << "블록2 isComputed 예외: " << e.what() << std::endl;
    }
    try {
        block2->getLastAccessed();
    } catch (const std::runtime_error& e) {
        std::cout << "블록2 getLastAccessed 예외: " << e.what() << std::endl;
    }

    // allocate_immutable_blocks 테스트
    std::vector<std::vector<int>> token_lists = { {9, 10}, {11, 12, 13} };
    std::vector<std::shared_ptr<Block>> blocks = allocator.allocate_immutable_blocks(block2, token_lists);
    std::cout << "\n[allocate_immutable_blocks 테스트]" << std::endl;
    std::cout << "생성된 불변 블록 수: " << blocks.size() << std::endl;
    for (size_t i = 0; i < blocks.size(); ++i) {
        std::cout << "불변 블록 " << i << " ID: " << blocks[i]->getBlockId()
                  << ", 토큰 수: " << blocks[i]->getTokenIds().size() << std::endl;
    }

    // fork 테스트: block2를 기반으로 fork 변환
    std::vector<std::shared_ptr<Block>> forked = allocator.fork(block2);
    std::cout << "\n[fork 테스트]" << std::endl;
    if (!forked.empty()) {
        std::cout << "Fork된 블록 ID: " << forked[0]->getBlockId() << std::endl;
    }

    // swap_out 및 swap_in 테스트
    std::cout << "\n[swap_out 및 swap_in 테스트]" << std::endl;
    allocator.swap_out(blocks);
    allocator.swap_in(blocks);
    if (!blocks.empty()) {
        std::cout << "swap_in 후 첫번째 블록 ID: " << blocks[0]->getBlockId() << std::endl;
    }

    // 추가적인 깊이 있는 기능 테스트: mark, clear, common, 물리적 ID, COW, immutable 승격, 등
    std::cout << "\n[추가 기능 테스트]" << std::endl;
    std::vector<int> test_block_ids = { block1->getBlockId(), block2->getBlockId() };
    allocator.mark_blocks_as_accessed(test_block_ids, 12345678.0);
    allocator.mark_blocks_as_computed(test_block_ids);
    std::vector<std::pair<int, int>> cow_info = allocator.clear_copy_on_writes();
    std::cout << "clear_copy_on_writes 결과, 반환 개수: " << cow_info.size() << std::endl;

    std::vector<std::vector<int>> computed_seq = { {block1->getBlockId()}, {block2->getBlockId()} };
    std::vector<int> common_computed = allocator.get_common_computed_block_ids(computed_seq);
    std::cout << "get_common_computed_block_ids 결과, 개수: " << common_computed.size() << std::endl;

    int physical_id = allocator.get_physical_block_id(block1->getBlockId());
    std::cout << "블록1의 물리적 ID: " << physical_id << std::endl;

    int cow_block = allocator.cow_block_if_not_appendable(*block1);
    std::cout << "cow_block_if_not_appendable 결과: " << cow_block << std::endl;

    int promoted = allocator.promote_to_immutable_block(*block1);
    std::cout << "promote_to_immutable_block 결과: " << promoted << std::endl;

    int full_touched = allocator.get_num_full_blocks_touched({block1, block2});
    std::cout << "get_num_full_blocks_touched 결과: " << full_touched << std::endl;

    float hit_rate = allocator.get_prefix_cache_hit_rate();
    std::cout << "get_prefix_cache_hit_rate 결과: " << hit_rate << std::endl;

    bool reset_result = allocator.reset_prefix_cache();
    std::cout << "reset_prefix_cache 결과: " << (reset_result ? "true" : "false") << std::endl;

    std::vector<int> cached_prefix = allocator.find_cached_blocks_prefix({ block1->getContentHash(), block2->getContentHash() });
    std::cout << "find_cached_blocks_prefix 반환 개수: " << cached_prefix.size() << std::endl;

    // free 함수 테스트: block1을 해제 후 남은 블록 수 확인
    std::cout << "\n[free 함수 테스트]" << std::endl;
    allocator.free(block1.get());
    std::cout << "block1 해제 후 남은 블록 수: " << allocator.get_num_free_blocks() << std::endl;

    return 0;
}