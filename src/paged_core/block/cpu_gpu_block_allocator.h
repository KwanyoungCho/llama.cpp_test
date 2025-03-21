#ifndef CPU_GPU_BLOCK_ALLOCATOR_H
#define CPU_GPU_BLOCK_ALLOCATOR_H

#include "common.h"
#include "interfaces.h"
#include "naive_block.h"
#include "block_table.h"

#include <vector>
#include <memory>
#include <deque>
#include <stdexcept>
#include <optional>
#include <unordered_set>
#include <unordered_map>

// CPUGPUBlockAllocator는 DeviceAwareBlockAllocator 인터페이스를 구현합니다.
class CPUGPUBlockAllocator : public DeviceAwareBlockAllocator {
public:
    CPUGPUBlockAllocator(int numBlocks, int blockSize);
    
    // DeviceAwareBlockAllocator 인터페이스 구현
    std::shared_ptr<Block> allocate_mutable_block(std::shared_ptr<Block> prevBlock, Device device) override;
    std::shared_ptr<Block> allocate_immutable_block(std::shared_ptr<Block> prevBlock, const std::vector<int>& tokenIds, Device device) override;
    std::vector<std::shared_ptr<Block>> allocate_immutable_blocks(std::shared_ptr<Block> prevBlock, const std::vector<std::vector<int>>& block_token_ids, Device device) override;
    
    int get_num_free_blocks(Device device) override;
    int get_num_total_blocks(Device device) override;
    
    void free(Block* block) override;
    std::vector<std::shared_ptr<Block>> fork(std::shared_ptr<Block> lastBlock) override;
    const std::unordered_set<int>& all_block_ids() const override;
    std::vector<std::pair<int, int>> clear_copy_on_writes() override;
    void mark_blocks_as_accessed(const std::vector<int>& block_ids, double now) override;
    void mark_blocks_as_computed(const std::vector<int>& block_ids) override;
    std::vector<int> get_common_computed_block_ids(const std::vector<std::vector<int>>& computed_seq_block_ids) override;
    int get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& blocks, Device device) override;
    std::unordered_map<int, int> swap(const std::vector<std::shared_ptr<Block>>& blocks, Device src_device, Device dst_device) override;
    int get_physical_block_id(Device device, int absolute_id) override;
    std::shared_ptr<Block> allocate_or_get_null_block() override;
    float get_prefix_cache_hit_rate(Device device) override;
    bool reset_prefix_cache() override;
    std::vector<int> find_cached_blocks_prefix(const std::vector<int>& block_hashes, Device device = Device::GPU) override;

private:
    int allocateBlockId();
    void freeBlockId(int blockId);
    static std::vector<int> generateBlockIndices(int numBlocks);
    static std::deque<int> createFreeBlockIds(int numBlocks);
    
    int numBlocks_;
    int blockSize_;
    RefCounter refCounter_;
    CopyOnWriteTracker cowTracker_;
    BlockPool blockPool_;
    std::deque<int> freeBlockIds_;
};

#endif // CPU_GPU_BLOCK_ALLOCATOR_H 