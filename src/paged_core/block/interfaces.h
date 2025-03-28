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

using BlockId = int;

// Forward declaration for BlockAllocator
class BlockAllocator;

// Abstract interface for a Block used in paged attention
class Block {
public:
    virtual ~Block() = default;

    virtual void append_token_ids(const std::vector<int>& token_ids) = 0;
    
    virtual int block_id() const = 0;
    virtual void set_block_id(int block_id) = 0;
    
    virtual const std::vector<int>& token_ids() const = 0;
    virtual std::vector<int>& token_ids() = 0;
    
    virtual int num_tokens_total() const = 0;
    virtual int num_empty_slots() const = 0;
    virtual bool is_full() const = 0;
    
    virtual std::shared_ptr<Block> prev_block() const = 0;
    virtual void set_prev_block(std::shared_ptr<Block> prev_block) = 0;
    
    virtual bool computed() const { throw std::runtime_error("computed not implemented"); }
    virtual void set_computed(bool value) { throw std::runtime_error("set_computed not implemented"); }
    
    virtual double last_accessed() const { throw std::runtime_error("last_accessed not implemented"); }
    virtual void set_last_accessed(double timestamp) { throw std::runtime_error("set_last_accessed not implemented"); }
    
    virtual int content_hash() const { return 0; }

    virtual int block_size() const = 0;

    // pool_id 관련 메서드 추가
    virtual int pool_id() const = 0;
    virtual void set_pool_id(int pool_id) = 0;
    
    // Factory interface for creating Block objects
    class Factory {
    public:
        virtual ~Factory() = default;
        virtual std::shared_ptr<Block> operator()(
            std::shared_ptr<Block> prev_block,
            const std::vector<int>& token_ids,
            int block_size,
            BlockAllocator* allocator,
            int block_id = -1,
            bool computed = false) = 0;
    };
};

// Abstract interface for a BlockAllocator
class BlockAllocator {
public:
    virtual ~BlockAllocator() = default;

    class NoFreeBlocksError : public std::runtime_error {
    public:
        NoFreeBlocksError() : std::runtime_error("No free blocks available") {}
    };

    virtual std::shared_ptr<Block> allocate_mutable_block(
        std::shared_ptr<Block> prev_block) = 0;

    virtual std::shared_ptr<Block> allocate_immutable_block(
        std::shared_ptr<Block> prev_block,
        const std::vector<int>& token_ids) = 0;

    virtual std::vector<std::shared_ptr<Block>> allocate_immutable_blocks(
        std::shared_ptr<Block> prev_block,
        const std::vector<std::vector<int>>& block_token_ids) = 0;

    virtual void free(std::shared_ptr<Block> block, bool keep_block_object = false) = 0;
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
    virtual std::vector<int> get_common_computed_block_ids(
        const std::vector<std::vector<int>>& computed_seq_block_ids) = 0;
    virtual BlockId cow_block_if_not_appendable(const Block& block) = 0;
    virtual BlockId promote_to_immutable_block(const Block& block) = 0;
    virtual int get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& blocks) = 0;
    virtual float get_prefix_cache_hit_rate() = 0;
    virtual bool reset_prefix_cache() = 0;
    virtual std::vector<int> find_cached_blocks_prefix(const std::vector<int>& block_hashes) = 0;
};

// Abstract interface for a DeviceAwareBlockAllocator
class DeviceAwareBlockAllocator {
public:
    virtual ~DeviceAwareBlockAllocator() = default;

    virtual std::shared_ptr<Block> allocate_mutable_block(
        std::shared_ptr<Block> prev_block,
        Device device) = 0;

    virtual std::shared_ptr<Block> allocate_immutable_block(
        std::shared_ptr<Block> prev_block,
        const std::vector<int>& token_ids,
        Device device) = 0;

    virtual std::vector<std::shared_ptr<Block>> allocate_immutable_blocks(
        std::shared_ptr<Block> prev_block,
        const std::vector<std::vector<int>>& block_token_ids,
        Device device) = 0;

    virtual int get_num_free_blocks(Device device) = 0;
    virtual int get_num_total_blocks(Device device) = 0;
    virtual void free(Block* block) = 0;
    virtual std::vector<std::shared_ptr<Block>> fork(std::shared_ptr<Block> last_block) = 0;
    virtual const std::unordered_set<int>& all_block_ids() const = 0;
    virtual std::vector<std::pair<int, int>> clear_copy_on_writes() = 0;
    virtual void mark_blocks_as_accessed(const std::vector<int>& block_ids, double now) = 0;
    virtual void mark_blocks_as_computed(const std::vector<int>& block_ids) = 0;
    virtual std::vector<int> get_common_computed_block_ids(
        const std::vector<std::vector<int>>& computed_seq_block_ids) = 0;
    virtual int get_num_full_blocks_touched(
        const std::vector<std::shared_ptr<Block>>& blocks,
        Device device) = 0;
    virtual std::unordered_map<int, int> swap(
        const std::vector<std::shared_ptr<Block>>& blocks,
        Device src_device,
        Device dst_device) = 0;
    virtual int get_physical_block_id(Device device, int absolute_id) = 0;
    virtual std::shared_ptr<Block> allocate_or_get_null_block() = 0;
    virtual float get_prefix_cache_hit_rate(Device device) = 0;
    virtual bool reset_prefix_cache() = 0;
    virtual std::vector<int> find_cached_blocks_prefix(
        const std::vector<int>& block_hashes,
        Device device = Device::GPU) = 0;
};

#endif // BLOCK_INTERFACES_H
