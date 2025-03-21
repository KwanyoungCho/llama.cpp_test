#ifndef BLOCK_INTERFACES_H
#define BLOCK_INTERFACES_H

#include <vector>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>

// Forward declaration for Device enum
enum class Device {
    CPU,
    GPU
};

// Abstract interface for a Block used in paged attention
class Block {
public:
    virtual ~Block() = default;
    virtual void appendTokenIds(const std::vector<int>& tokenIds) = 0;
    virtual int getBlockId() const = 0;
    virtual void setBlockId(int id) = 0;
    virtual std::vector<int>& getTokenIds() = 0;
    virtual int numEmptySlots() const = 0;
    virtual bool isFull() const = 0;
    virtual std::shared_ptr<Block> getPrevBlock() const = 0;
    
    // Additional functionalities (paralleling the Python interface)
    virtual int numTokensTotal() const = 0;
    virtual bool isComputed() const { throw std::runtime_error("Block::isComputed not implemented"); }
    virtual void setComputed(bool) { throw std::runtime_error("Block::setComputed not implemented"); }
    virtual double getLastAccessed() const { throw std::runtime_error("Block::getLastAccessed not implemented"); }
    virtual void setLastAccessed(double) { throw std::runtime_error("Block::setLastAccessed not implemented"); }
    virtual int getContentHash() const { return 0; }
};

// Abstract interface for a BlockAllocator
class BlockAllocator {
public:
    virtual ~BlockAllocator() = default;
    virtual std::shared_ptr<Block> allocate_mutable_block(std::shared_ptr<Block> prev_block) = 0;
    virtual std::shared_ptr<Block> allocate_immutable_block(std::shared_ptr<Block> prev_block, const std::vector<int>& token_ids) = 0;
    virtual std::vector<std::shared_ptr<Block>> allocate_immutable_blocks(std::shared_ptr<Block> prev_block, const std::vector<std::vector<int>>& block_token_ids) = 0;
    virtual void free(Block* block) = 0;
    virtual std::vector<std::shared_ptr<Block>> fork(std::shared_ptr<Block> last_block) = 0;
    virtual int get_num_total_blocks() const = 0;
    virtual int get_num_free_blocks() const = 0;
    virtual int get_physical_block_id(int absolute_id) = 0;
    virtual void swap_out(const std::vector<std::shared_ptr<Block>>& blocks) = 0;
    virtual void swap_in(const std::vector<std::shared_ptr<Block>>& blocks) = 0;
    virtual const std::unordered_set<int>& all_block_ids() const = 0;
    virtual std::vector<std::pair<int, int>> clear_copy_on_writes() = 0;
    virtual void mark_blocks_as_accessed(const std::vector<int>& block_ids, double now) = 0;
    virtual void mark_blocks_as_computed(const std::vector<int>& block_ids) = 0;
    virtual std::vector<int> get_common_computed_block_ids(const std::vector<std::vector<int>>& computed_seq_block_ids) = 0;
    virtual int cow_block_if_not_appendable(const Block& block) = 0;
    virtual int promote_to_immutable_block(const Block& block) = 0;
    virtual int get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& blocks) = 0;
    virtual float get_prefix_cache_hit_rate() = 0;
    virtual bool reset_prefix_cache() = 0;
    virtual std::vector<int> find_cached_blocks_prefix(const std::vector<int>& block_hashes) = 0;
};

// Abstract interface for a DeviceAwareBlockAllocator
class DeviceAwareBlockAllocator {
public:
    virtual ~DeviceAwareBlockAllocator() = default;
    virtual std::shared_ptr<Block> allocate_mutable_block(std::shared_ptr<Block> prev_block, Device device) = 0;
    virtual std::shared_ptr<Block> allocate_immutable_block(std::shared_ptr<Block> prev_block, const std::vector<int>& token_ids, Device device) = 0;
    virtual std::vector<std::shared_ptr<Block>> allocate_immutable_blocks(std::shared_ptr<Block> prev_block, const std::vector<std::vector<int>>& block_token_ids, Device device) = 0;
    virtual int get_num_free_blocks(Device device) = 0;
    virtual int get_num_total_blocks(Device device) = 0;
    virtual void free(Block* block) = 0;
    virtual std::vector<std::shared_ptr<Block>> fork(std::shared_ptr<Block> last_block) = 0;
    virtual const std::unordered_set<int>& all_block_ids() const = 0;
    virtual std::vector<std::pair<int, int>> clear_copy_on_writes() = 0;
    virtual void mark_blocks_as_accessed(const std::vector<int>& block_ids, double now) = 0;
    virtual void mark_blocks_as_computed(const std::vector<int>& block_ids) = 0;
    virtual std::vector<int> get_common_computed_block_ids(const std::vector<std::vector<int>>& computed_seq_block_ids) = 0;
    virtual int get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& blocks, Device device) = 0;
    virtual std::unordered_map<int, int> swap(const std::vector<std::shared_ptr<Block>>& blocks, Device src_device, Device dst_device) = 0;
    virtual int get_physical_block_id(Device device, int absolute_id) = 0;
    virtual std::shared_ptr<Block> allocate_or_get_null_block() = 0;
    virtual float get_prefix_cache_hit_rate(Device device) = 0;
    virtual bool reset_prefix_cache() = 0;
    virtual std::vector<int> find_cached_blocks_prefix(const std::vector<int>& block_hashes, Device device = Device::GPU) = 0;
};

#endif // BLOCK_INTERFACES_H 