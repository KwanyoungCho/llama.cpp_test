#include "common.h"
#include <algorithm>
#include <functional>
#include <stdexcept>

// ------------------- RefCounter -------------------

RefCounter::RefCounter(const std::vector<int>& allBlockIndices) {
    for (int id : allBlockIndices) {
        refcounts_[id] = 0;
    }
}

int RefCounter::incr(int blockId) {
    auto it = refcounts_.find(blockId);
    if (it == refcounts_.end()) {
        throw std::runtime_error("BlockId not found in RefCounter::incr");
    }
    int pre = it->second;
    if (pre < 0) {
        throw std::runtime_error("Negative refcount in RefCounter::incr");
    }
    refcounts_[blockId] = pre + 1;
    return refcounts_[blockId];
}

int RefCounter::decr(int blockId) {
    auto it = refcounts_.find(blockId);
    if (it == refcounts_.end()) {
        throw std::runtime_error("BlockId not found in RefCounter::decr");
    }
    int count = it->second;
    if (count <= 0) {
        throw std::runtime_error("Refcount must be greater than 0 in RefCounter::decr");
    }
    refcounts_[blockId] = count - 1;
    return refcounts_[blockId];
}

int RefCounter::get(int blockId) const {
    auto it = refcounts_.find(blockId);
    if (it == refcounts_.end()) {
        throw std::runtime_error("BlockId not found in RefCounter::get");
    }
    return it->second;
}

// ------------------- ReadOnlyRefCounter -------------------

ReadOnlyRefCounter::ReadOnlyRefCounter(const RefCounter& refcounter) : refcounter_(refcounter) {}

int ReadOnlyRefCounter::get(int blockId) const {
    return refcounter_.get(blockId);
}

int ReadOnlyRefCounter::incr(int) const {
    throw std::runtime_error("Incr not allowed in ReadOnlyRefCounter");
}

int ReadOnlyRefCounter::decr(int) const {
    throw std::runtime_error("Decr not allowed in ReadOnlyRefCounter");
}

// ------------------- CopyOnWriteTracker -------------------

CopyOnWriteTracker::CopyOnWriteTracker(const RefCounter& refcounter) : refCounter_(refcounter) {}

bool CopyOnWriteTracker::isAppendable(const std::shared_ptr<Block>& block) const {
    int id = block->getBlockId();
    // If blockId is negative, treat as unallocated/appendable
    if (id < 0) {
        return true;
    }
    return refCounter_.get(id) <= 1;
}

void CopyOnWriteTracker::recordCow(int srcBlockId, int trgBlockId) {
    if (srcBlockId < 0 || trgBlockId < 0) {
        throw std::runtime_error("Invalid blockId in recordCow");
    }
    copyOnWrites_.emplace_back(srcBlockId, trgBlockId);
}

std::vector<std::pair<int, int>> CopyOnWriteTracker::clearCows() {
    std::vector<std::pair<int, int>> cows = copyOnWrites_;
    copyOnWrites_.clear();
    return cows;
}

// ------------------- BlockPool -------------------

BlockPool::BlockPool(int blockSize, BlockFactory create_block, int poolSize)
    : blockSize_(blockSize), create_block_(create_block), poolSize_(poolSize) {}

std::shared_ptr<Block> BlockPool::initBlock(std::shared_ptr<Block> prevBlock,
                                              const std::vector<int>& tokenIds,
                                              int blockSize,
                                              int physicalBlockId) {
    auto block = create_block_(prevBlock, blockSize);
    block->setBlockId(physicalBlockId);
    block->appendTokenIds(tokenIds);
    return block;
}

void BlockPool::freeBlock(const std::shared_ptr<Block>& block) {
    // No pooling logic implemented in this minimal version
}

// ------------------- BlockList -------------------

BlockList::BlockList() {}

void BlockList::update(const std::vector<std::shared_ptr<Block>>& blocks) {
    _blocks = blocks;
    _block_ids.clear();
    for (auto& block : _blocks) {
        if (block->getBlockId() < 0) {
            throw std::runtime_error("BlockList::update found negative blockId");
        }
        _block_ids.push_back(block->getBlockId());
    }
}

void BlockList::append_token_ids(int block_index, const std::vector<int>& token_ids) {
    if (block_index < 0 || block_index >= static_cast<int>(_blocks.size())) {
        throw std::runtime_error("Invalid block index in BlockList::append_token_ids");
    }
    auto& block = _blocks[block_index];
    int prev_id = block->getBlockId();
    block->appendTokenIds(token_ids);
    if (prev_id != block->getBlockId()) {
        _block_ids[block_index] = block->getBlockId();
    }
}

void BlockList::append(const std::shared_ptr<Block>& block) {
    _blocks.push_back(block);
    _block_ids.push_back(block->getBlockId());
}

size_t BlockList::size() const {
    return _blocks.size();
}

std::shared_ptr<Block>& BlockList::operator[](size_t index) {
    return _blocks[index];
}

const std::vector<std::shared_ptr<Block>>& BlockList::list() const {
    return _blocks;
}

const std::vector<int>& BlockList::ids() const {
    return _block_ids;
}

void BlockList::reset() {
    _blocks.clear();
    _block_ids.clear();
}

// ------------------- CacheMetricData -------------------

void CacheMetricData::query(bool hit) {
    num_incompleted_block_queries += 1;
    if (hit) {
        num_incompleted_block_hit += 1;
    }
    if (num_incompleted_block_queries == blockSize) {
        float hit_rate = static_cast<float>(num_incompleted_block_hit) / num_incompleted_block_queries;
        completed_block_cache_hit_rate = (completed_block_cache_hit_rate * num_completed_blocks + hit_rate) / (num_completed_blocks + 1);
        num_incompleted_block_queries = 0;
        num_incompleted_block_hit = 0;
        num_completed_blocks += 1;
    }
}

float CacheMetricData::get_hit_rate() const {
    float incomplete_ratio = static_cast<float>(num_incompleted_block_queries) / blockSize;
    float total_blocks = num_completed_blocks + incomplete_ratio;
    if (total_blocks == 0) return 0.0f;
    float completed_block_hit = (num_completed_blocks > 0) ? (completed_block_cache_hit_rate * num_completed_blocks) : 0.0f;
    float incompleted_block_hit = (num_incompleted_block_queries > 0)
        ? ((static_cast<float>(num_incompleted_block_hit) / num_incompleted_block_queries) * incomplete_ratio)
        : 0.0f;
    return (completed_block_hit + incompleted_block_hit) / total_blocks;
}

// ------------------- getAllBlocksRecursively -------------------

std::vector<std::shared_ptr<Block>> getAllBlocksRecursively(const std::shared_ptr<Block>& lastBlock) {
    std::vector<std::shared_ptr<Block>> allBlocks;
    std::function<void(const std::shared_ptr<Block>&)> recurse = [&](const std::shared_ptr<Block>& block) {
        if (block->getPrevBlock() != nullptr) {
            recurse(block->getPrevBlock());
        }
        allBlocks.push_back(block);
    };
    recurse(lastBlock);
    return allBlocks;
} 