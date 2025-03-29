#include "cpu_gpu_block_allocator.h"
#include <stdexcept>

namespace paged_core {

std::unique_ptr<CpuGpuBlockAllocator> CpuGpuBlockAllocator::create(
    const std::string& allocator_type,
    int num_gpu_blocks,
    int num_cpu_blocks,
    int block_size) {
    // TODO: Implement different allocator types (naive, prefix_caching)
    // For now, we'll just create a basic implementation
    throw std::runtime_error("Not implemented yet");
}

CpuGpuBlockAllocator::CpuGpuBlockAllocator(
    std::unique_ptr<BlockAllocator> cpu_allocator,
    std::unique_ptr<BlockAllocator> gpu_allocator)
    : _cpu_allocator(std::move(cpu_allocator))
    , _gpu_allocator(std::move(gpu_allocator)) {
    // Initialize block ID mappings
    // TODO: Implement block ID mapping initialization
}

Block* CpuGpuBlockAllocator::allocate_mutable_block(Block* prev_block, Device device) {
    BlockAllocator* allocator = (device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    return allocator->allocate_mutable_block(prev_block);
}

std::vector<Block*> CpuGpuBlockAllocator::allocate_immutable_blocks(
    Block* prev_block,
    const std::vector<std::vector<int>>& block_token_ids,
    Device device) {
    BlockAllocator* allocator = (device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    return allocator->allocate_immutable_blocks(prev_block, block_token_ids);
}

void CpuGpuBlockAllocator::free(Block* block) {
    if (!block) return;
    
    int block_id = block->get_block_id();
    auto it = _block_ids_to_allocator.find(block_id);
    if (it != _block_ids_to_allocator.end()) {
        it->second->free(block);
    }
}

std::vector<Block*> CpuGpuBlockAllocator::fork(Block* last_block) {
    if (!last_block) return {};
    
    int block_id = last_block->get_block_id();
    auto it = _block_ids_to_allocator.find(block_id);
    if (it != _block_ids_to_allocator.end()) {
        return it->second->fork(last_block);
    }
    return {};
}

int CpuGpuBlockAllocator::get_num_free_blocks(Device device) const {
    BlockAllocator* allocator = (device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    return allocator->get_num_free_blocks();
}

int CpuGpuBlockAllocator::get_num_total_blocks(Device device) const {
    BlockAllocator* allocator = (device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    return allocator->get_num_total_blocks();
}

int CpuGpuBlockAllocator::get_physical_block_id(Device device, int absolute_id) const {
    BlockAllocator* allocator = (device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    return allocator->get_physical_block_id(absolute_id);
}

std::unordered_map<int, int> CpuGpuBlockAllocator::swap(
    const std::vector<Block*>& blocks,
    Device src_device,
    Device dst_device) {
    
    BlockAllocator* src_allocator = (src_device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    BlockAllocator* dst_allocator = (dst_device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    
    std::vector<int> src_block_ids;
    for (Block* block : blocks) {
        if (block) {
            src_block_ids.push_back(block->get_block_id());
        }
    }
    
    src_allocator->swap_out(blocks);
    dst_allocator->swap_in(blocks);
    
    std::unordered_map<int, int> current_swap_mapping;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i]) {
            int dst_block_id = blocks[i]->get_block_id();
            if (dst_block_id != -1) {
                _swap_mapping[src_block_ids[i]] = dst_block_id;
                current_swap_mapping[src_block_ids[i]] = dst_block_id;
            }
        }
    }
    
    return current_swap_mapping;
}

int CpuGpuBlockAllocator::get_num_full_blocks_touched(
    const std::vector<Block*>& blocks,
    Device device) const {
    
    BlockAllocator* allocator = (device == Device::CPU) ? _cpu_allocator.get() : _gpu_allocator.get();
    return allocator->get_num_full_blocks_touched(blocks);
}

} // namespace paged_core
