#include "cpu_gpu_block_allocator.h"

#include <stdexcept>
#include <deque>
#include <vector>
#include <optional>

CPUGPUBlockAllocator::CPUGPUBlockAllocator(int numBlocks, int blockSize)
    : numBlocks_(numBlocks),
      blockSize_(blockSize),
      refCounter_(generateBlockIndices(numBlocks)),
      cowTracker_(refCounter_),
      blockPool_(blockSize, [](std::shared_ptr<Block> prev, int size) {
          return std::make_shared<NaiveBlock>(prev, size);
      }, numBlocks * 4),
      freeBlockIds_(createFreeBlockIds(numBlocks))
{
}

std::shared_ptr<Block> CPUGPUBlockAllocator::allocate_mutable_block(std::shared_ptr<Block> prevBlock, Device /*device*/, std::optional<int> /*extra_hash*/) {
    int blockId = allocateBlockId();
    auto block = blockPool_.initBlock(prevBlock, std::vector<int>{}, blockSize_, blockId);
    return block;
}

std::shared_ptr<Block> CPUGPUBlockAllocator::allocate_immutable_block(std::shared_ptr<Block> prevBlock, const std::vector<int>& tokenIds, Device device, std::optional<int> extra_hash) {
    auto block = allocate_mutable_block(prevBlock, device, extra_hash);
    block->appendTokenIds(tokenIds);
    return block;
}

void CPUGPUBlockAllocator::free(Block* block) {
    if (!block) {
        throw std::runtime_error("Null block cannot be freed");
    }
    int id = block->getBlockId();
    if (id < 0) {
        throw std::runtime_error("Block is not allocated");
    }
    freeBlockId(id);
    blockPool_.freeBlock(std::shared_ptr<Block>(block));
}

int CPUGPUBlockAllocator::get_num_free_blocks(Device /*device*/) {
    return static_cast<int>(freeBlockIds_.size());
}

int CPUGPUBlockAllocator::get_num_total_blocks(Device /*device*/) {
    return numBlocks_;
}

int CPUGPUBlockAllocator::allocateBlockId() {
    if (freeBlockIds_.empty()) {
        throw std::runtime_error("No free blocks available");
    }
    int id = freeBlockIds_.front();
    freeBlockIds_.pop_front();
    refCounter_.incr(id);
    return id;
}

void CPUGPUBlockAllocator::freeBlockId(int blockId) {
    if (refCounter_.decr(blockId) == 0) {
        freeBlockIds_.push_back(blockId);
    }
}

std::vector<int> CPUGPUBlockAllocator::generateBlockIndices(int numBlocks) {
    std::vector<int> indices;
    for (int i = 0; i < numBlocks; ++i) {
        indices.push_back(i);
    }
    return indices;
}

std::deque<int> CPUGPUBlockAllocator::createFreeBlockIds(int numBlocks) {
    std::deque<int> dq;
    for (int i = 0; i < numBlocks; ++i) {
        dq.push_back(i);
    }
    return dq;
}

std::vector<std::shared_ptr<Block>> CPUGPUBlockAllocator::allocate_immutable_blocks(std::shared_ptr<Block> prevBlock, const std::vector<std::vector<int>>& block_token_ids, Device device, std::optional<int> extra_hash) {
    std::vector<std::shared_ptr<Block>> blocks;
    for (const auto& tokenIds : block_token_ids) {
        blocks.push_back(allocate_immutable_block(prevBlock, tokenIds, device, extra_hash));
    }
    return blocks;
}

std::vector<std::shared_ptr<Block>> CPUGPUBlockAllocator::fork(std::shared_ptr<Block> /*lastBlock*/) {
    throw std::runtime_error("fork not implemented in CPUGPUBlockAllocator");
}

const std::unordered_set<int>& CPUGPUBlockAllocator::all_block_ids() const {
    throw std::runtime_error("all_block_ids not implemented in CPUGPUBlockAllocator");
}

std::vector<std::pair<int, int>> CPUGPUBlockAllocator::clear_copy_on_writes() {
    throw std::runtime_error("clear_copy_on_writes not implemented in CPUGPUBlockAllocator");
}

void CPUGPUBlockAllocator::mark_blocks_as_accessed(const std::vector<int>& /*block_ids*/, double /*now*/) {
    throw std::runtime_error("mark_blocks_as_accessed not implemented in CPUGPUBlockAllocator");
}

void CPUGPUBlockAllocator::mark_blocks_as_computed(const std::vector<int>& /*block_ids*/) {
    throw std::runtime_error("mark_blocks_as_computed not implemented in CPUGPUBlockAllocator");
}

std::vector<int> CPUGPUBlockAllocator::get_common_computed_block_ids(const std::vector<std::vector<int>>& /*computed_seq_block_ids*/) {
    throw std::runtime_error("get_common_computed_block_ids not implemented in CPUGPUBlockAllocator");
}

int CPUGPUBlockAllocator::get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& /*blocks*/, Device /*device*/) {
    throw std::runtime_error("get_num_full_blocks_touched not implemented in CPUGPUBlockAllocator");
}

std::unordered_map<int, int> CPUGPUBlockAllocator::swap(const std::vector<std::shared_ptr<Block>>& /*blocks*/, Device /*src_device*/, Device /*dst_device*/) {
    throw std::runtime_error("swap not implemented in CPUGPUBlockAllocator");
}

int CPUGPUBlockAllocator::get_physical_block_id(Device /*device*/, int absolute_id) {
    return absolute_id;
}

std::shared_ptr<Block> CPUGPUBlockAllocator::allocate_or_get_null_block() {
    return nullptr;
}

float CPUGPUBlockAllocator::get_prefix_cache_hit_rate(Device /*device*/) {
    return 0.0f;
}

bool CPUGPUBlockAllocator::reset_prefix_cache() {
    return false;
}

std::vector<int> CPUGPUBlockAllocator::find_cached_blocks_prefix(const std::vector<int>& /*block_hashes*/, Device /*device*/) {
    throw std::runtime_error("find_cached_blocks_prefix not implemented in CPUGPUBlockAllocator");
} 