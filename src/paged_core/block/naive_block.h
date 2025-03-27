#ifndef NAIVE_BLOCK_H
#define NAIVE_BLOCK_H

#include "interfaces.h"
#include "common.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <deque>
#include <set>
#include <unordered_set>
#include <unordered_map>

// 전방 선언: BlockAllocator
// class BlockAllocator;

class NaiveBlock : public PooledBlock {
public:
    NaiveBlock(std::shared_ptr<Block> prev_block, 
               const std::vector<int>& token_ids,
               int block_size,
               BlockAllocator* allocator,
               int block_id = -1,
               std::shared_ptr<Block> cow_target = nullptr);
    
    ~NaiveBlock() override = default;
    
    // Block 인터페이스 구현
    void append_token_ids(const std::vector<int>& token_ids) override;
    const std::vector<int>& token_ids() const override;
    std::vector<int>& token_ids() override;
    bool is_full() const override;
    int num_empty_slots() const override;
    std::shared_ptr<Block> prev_block() const override;
    void set_prev_block(std::shared_ptr<Block> prev_block) override;
    int block_id() const override;
    void set_block_id(int block_id) override;
    bool computed() const override;
    void set_computed(bool value) override;
    double last_accessed() const override;
    void set_last_accessed(double timestamp) override;
    int content_hash() const override;
    int block_size() const override;
    int num_tokens_total() const override { 
        throw std::runtime_error("num_tokens_total is not used for naive block"); 
    }
    
    // PooledBlock 인터페이스 구현
    int pool_id() const override;
    void set_pool_id(int pool_id) override;

private:
    void _append_token_ids_no_cow(const std::vector<int>& token_ids);

    std::vector<int> _token_ids;
    std::shared_ptr<Block> _prev_block;
    int _block_size;
    BlockAllocator* _allocator;
    int _block_id;
    int _pool_id = -1;  // pool_id 멤버 변수 추가
    std::shared_ptr<Block> _cow_target;
    // bool _computed;
    // double _last_accessed;
    // mutable int _content_hash;
};

class NaiveBlockAllocator : public BlockAllocator {
public:
    NaiveBlockAllocator(std::shared_ptr<Block::Factory> create_block,
                       int num_blocks,
                       int block_size,
                       const std::vector<int>& block_ids = std::vector<int>(),
                       std::shared_ptr<BlockPool> block_pool = nullptr);

    // BlockAllocator 인터페이스 구현
    std::shared_ptr<Block> allocate_mutable_block(std::shared_ptr<Block> prev_block) override;
    std::shared_ptr<Block> allocate_immutable_block(std::shared_ptr<Block> prev_block,
                                                   const std::vector<int>& token_ids) override;
    std::vector<std::shared_ptr<Block>> allocate_immutable_blocks(
        std::shared_ptr<Block> prev_block,
        const std::vector<std::vector<int>>& block_token_ids) override;
    void free(std::shared_ptr<Block> block, bool keep_block_object = false) override;
    void free_block_id(int block_id);
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
    std::vector<int> get_common_computed_block_ids(
        const std::vector<std::vector<int>>& computed_seq_block_ids) override;
    int cow_block_if_not_appendable(const Block& block) override;
    int promote_to_immutable_block(const Block& block) override;
    int get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& blocks) override;
    float get_prefix_cache_hit_rate() override { return -1.0f; }
    bool reset_prefix_cache() override { return true; }
    std::vector<int> find_cached_blocks_prefix(const std::vector<int>& block_hashes) override {
        return std::vector<int>();
    }

    const RefCounter& refcounter() const { return *_ref_counter; }
    int block_size() const { return _block_size; }

private:
    int _allocate_block_id();
    void _free_block_id(std::shared_ptr<Block> block);
    void _free_block_id(int block_id);

    std::shared_ptr<Block::Factory> _create_block;
    int _block_size;
    std::deque<int> _free_block_indices;
    std::unordered_set<int> _all_block_indices;
    std::unique_ptr<BlockPool> _block_pool;
    std::unique_ptr<CopyOnWriteTracker> _cow_tracker;
    std::unique_ptr<RefCounter> _ref_counter;
};

#endif // NAIVE_BLOCK_H