#ifndef NAIVE_BLOCK_H
#define NAIVE_BLOCK_H

#include "interfaces.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <deque>
#include <set>
#include <unordered_set>

// 전방 선언: BlockAllocator
class BlockAllocator;

class NaiveBlock : public Block, public std::enable_shared_from_this<NaiveBlock> {
public:
    // 생성자 및 소멸자
    NaiveBlock(std::shared_ptr<Block> prev_block, int block_size, BlockAllocator* allocator,
               const std::vector<int>& initial_token_ids = std::vector<int>(),
               int block_id = -1,
               std::shared_ptr<Block> cow_target = nullptr);
    ~NaiveBlock() override;

    // Block 인터페이스 구현 (인터페이스와 동일한 함수명 및 시그니처)
    void appendTokenIds(const std::vector<int>& token_ids) override;
    int getBlockId() const override;
    void setBlockId(int id) override;
    std::vector<int>& getTokenIds() override;
    int numEmptySlots() const override;
    bool isFull() const override;
    std::shared_ptr<Block> getPrevBlock() const override;
    
    // 추가 기능 (필요시 구현 또는 예외 발생)
    int numTokensTotal() const override;
    bool isComputed() const override;
    void setComputed(bool value) override;
    double getLastAccessed() const override;
    void setLastAccessed(double timestamp) override;
    int getContentHash() const override;

private:
    // 내부 헬퍼 함수
    void _appendTokenIdsNoCOW(const std::vector<int>& token_ids);

    // 멤버 변수
    int _blockId;
    std::vector<int> _tokenIds;
    std::shared_ptr<Block> _prevBlock;
    int _blockSize;
    BlockAllocator* _allocator;
    std::shared_ptr<Block> _cowTarget; // copy-on-write 대상
};

class NaiveBlockAllocator : public BlockAllocator {
public:
    // 생성자
    NaiveBlockAllocator(std::shared_ptr<Block::Factory> create_block, int num_blocks, int block_size, const std::vector<int>& block_ids = std::vector<int>());

    // BlockAllocator 인터페이스 구현 (함수명을 인터페이스와 동일하게)
    std::shared_ptr<Block> allocate_mutable_block(std::shared_ptr<Block> prev_block) override;
    std::shared_ptr<Block> allocate_immutable_block(std::shared_ptr<Block> prev_block, const std::vector<int>& token_ids) override;
    std::vector<std::shared_ptr<Block>> allocate_immutable_blocks(std::shared_ptr<Block> prev_block, const std::vector<std::vector<int>>& block_token_ids) override;
    void free(Block* block) override;
    std::vector<std::shared_ptr<Block>> fork(std::shared_ptr<Block> last_block) override;
    int get_num_total_blocks() const override;
    int get_num_free_blocks() const override;
    int get_physical_block_id(int absolute_id) override;
    void swap_out(const std::vector<std::shared_ptr<Block>>& blocks) override;
    void swap_in(const std::vector<std::shared_ptr<Block>>& blocks) override;
    const std::unordered_set<int>& all_block_ids() const override;
    std::vector<std::pair<int, int>> clear_copy_on_writes() override;
    void mark_blocks_as_accessed(const std::vector<int>& block_ids, double now) override;
    void mark_blocks_as_computed(const std::vector<int>& block_ids) override;
    std::vector<int> get_common_computed_block_ids(const std::vector<std::vector<int>>& computed_seq_block_ids) override;
    int cow_block_if_not_appendable(const Block& block) override;
    int promote_to_immutable_block(const Block& block) override;
    int get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& blocks) override;
    float get_prefix_cache_hit_rate() override;
    bool reset_prefix_cache() override;
    std::vector<int> find_cached_blocks_prefix(const std::vector<int>& block_hashes) override;

    // 추가: 블록 크기 getter
    int blockSize() const;

    // 멤버 변수
    std::shared_ptr<Block::Factory> createBlock;

private:
    int _allocateBlockId();
    void _freeBlockId(std::shared_ptr<Block> block);
    void _freeBlockId(int blockId);

    int _blockSize;
    std::deque<int> _freeBlockIndices;
    std::unordered_set<int> _allBlockIndices;
};

#endif // NAIVE_BLOCK_H