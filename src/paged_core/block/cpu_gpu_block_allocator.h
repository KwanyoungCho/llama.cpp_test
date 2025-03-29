#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>

namespace paged_core {

enum class Device {
    CPU,
    GPU
};

class Block {
public:
    virtual ~Block() = default;
    virtual int get_block_id() const = 0;
    virtual void set_block_id(int id) = 0;
    virtual std::vector<int> get_token_ids() const = 0;
    virtual int get_num_empty_slots() const = 0;
    virtual bool is_full() const = 0;
    virtual Block* get_prev_block() const = 0;
    virtual void set_prev_block(Block* block) = 0;
    virtual bool is_computed() const = 0;
    virtual void set_computed(bool computed) = 0;
    virtual float get_last_accessed() const = 0;
    virtual void set_last_accessed(float timestamp) = 0;
    virtual int get_content_hash() const = 0;
};

class BlockAllocator {
public:
    virtual ~BlockAllocator() = default;
    virtual Block* allocate_mutable_block(Block* prev_block = nullptr) = 0;
    virtual std::vector<Block*> allocate_immutable_blocks(
        Block* prev_block,
        const std::vector<std::vector<int>>& block_token_ids) = 0;
    virtual void free(Block* block) = 0;
    virtual std::vector<Block*> fork(Block* last_block) = 0;
    virtual int get_num_free_blocks() const = 0;
    virtual int get_num_total_blocks() const = 0;
    virtual int get_physical_block_id(int absolute_id) const = 0;
    virtual void swap_out(const std::vector<Block*>& blocks) = 0;
    virtual void swap_in(const std::vector<Block*>& blocks) = 0;
    virtual int get_num_full_blocks_touched(const std::vector<Block*>& blocks) const = 0;
};

class CpuGpuBlockAllocator {
public:
    static std::unique_ptr<CpuGpuBlockAllocator> create(
        const std::string& allocator_type,
        int num_gpu_blocks,
        int num_cpu_blocks,
        int block_size);

    CpuGpuBlockAllocator(
        std::unique_ptr<BlockAllocator> cpu_allocator,
        std::unique_ptr<BlockAllocator> gpu_allocator);

    Block* allocate_mutable_block(Block* prev_block, Device device);
    std::vector<Block*> allocate_immutable_blocks(
        Block* prev_block,
        const std::vector<std::vector<int>>& block_token_ids,
        Device device);
    void free(Block* block);
    std::vector<Block*> fork(Block* last_block);
    int get_num_free_blocks(Device device) const;
    int get_num_total_blocks(Device device) const;
    int get_physical_block_id(Device device, int absolute_id) const;
    std::unordered_map<int, int> swap(
        const std::vector<Block*>& blocks,
        Device src_device,
        Device dst_device);
    int get_num_full_blocks_touched(
        const std::vector<Block*>& blocks,
        Device device) const;

private:
    std::unique_ptr<BlockAllocator> _cpu_allocator;
    std::unique_ptr<BlockAllocator> _gpu_allocator;
    std::unordered_map<int, BlockAllocator*> _block_ids_to_allocator;
    std::unordered_map<int, int> _swap_mapping;
    std::optional<Block*> _null_block;
};

} // namespace paged_core
