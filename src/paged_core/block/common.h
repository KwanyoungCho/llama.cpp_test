#ifndef PAGED_CORE_BLOCK_COMMON_H
#define PAGED_CORE_BLOCK_COMMON_H

#include "interfaces.h"
#include <vector>
#include <deque>
#include <unordered_map>
#include <utility>
#include <functional>
#include <cassert>
#include <memory>
#include <stdexcept>

// RefCounter: manages reference counts for block IDs
class RefCounter {
public:
    // Initialize with all block indices
    RefCounter(const std::vector<int>& allBlockIndices) {
        for (int id : allBlockIndices) {
            refcounts_[id] = 0;
        }
    }

    // Increments reference count for blockId and returns new count
    int incr(int blockId) {
        assert(refcounts_.find(blockId) != refcounts_.end());
        int pre = refcounts_[blockId];
        assert(pre >= 0);
        refcounts_[blockId] = pre + 1;
        return refcounts_[blockId];
    }

    // Decrements reference count for blockId and returns new count
    int decr(int blockId) {
        assert(refcounts_.find(blockId) != refcounts_.end());
        int count = refcounts_[blockId];
        assert(count > 0);
        refcounts_[blockId] = count - 1;
        return refcounts_[blockId];
    }

    // Returns the reference count for blockId
    int get(int blockId) const {
        auto it = refcounts_.find(blockId);
        assert(it != refcounts_.end());
        return it->second;
    }

private:
    std::unordered_map<int, int> refcounts_;
};

// ReadOnlyRefCounter provides read-only access to RefCounter
class ReadOnlyRefCounter {
public:
    ReadOnlyRefCounter(const RefCounter& refcounter) : refcounter_(refcounter) {}
    
    int get(int blockId) const {
        return refcounter_.get(blockId);
    }
    
    // Disable modifications
    int incr(int) const {
        throw std::runtime_error("Incr not allowed in ReadOnlyRefCounter");
    }
    int decr(int) const {
        throw std::runtime_error("Decr not allowed in ReadOnlyRefCounter");
    }

private:
    const RefCounter& refcounter_;
};

// CopyOnWriteTracker tracks copy-on-write operations for blocks
class CopyOnWriteTracker {
public:
    // Construct with a reference to a RefCounter (read-only view expected)
    CopyOnWriteTracker(const RefCounter& refcounter) : refCounter_(refcounter) {}

    // isAppendable: returns true if the block is not shared
    bool isAppendable(const std::shared_ptr<Block>& block) const {
        int id = block->getBlockId();
        // If blockId is negative, treat it as unallocated/appendable
        if (id < 0) {
            return true;
        }
        return refCounter_.get(id) <= 1;
    }

    // Record a copy-on-write operation from srcBlockId to trgBlockId
    void recordCow(int srcBlockId, int trgBlockId) {
        assert(srcBlockId >= 0 && trgBlockId >= 0);
        copyOnWrites_.emplace_back(srcBlockId, trgBlockId);
    }

    // Clears and returns the current copy-on-write mappings
    std::vector<std::pair<int, int>> clearCows() {
        auto cows = copyOnWrites_;
        copyOnWrites_.clear();
        return cows;
    }

private:
    const RefCounter& refCounter_;
    std::vector<std::pair<int, int>> copyOnWrites_;
};

// Minimal BlockPool implementation
// This pool pre-allocates block objects to avoid excessive allocations.
// For simplicity, this minimal version does not reuse blocks from a pool.
class BlockPool {
public:
    // BlockFactory: function to create a new Block given a previous block and block size
    typedef std::function<std::shared_ptr<Block>(std::shared_ptr<Block> prevBlock, int blockSize)> BlockFactory;

    BlockPool(int blockSize, BlockFactory create_block, int poolSize)
        : blockSize_(blockSize), create_block_(create_block), poolSize_(poolSize) {}

    // Initializes a block: creates a block, sets its blockId, and appends tokenIds
    std::shared_ptr<Block> initBlock(std::shared_ptr<Block> prevBlock,
                                     const std::vector<int>& tokenIds,
                                     int blockSize,
                                     int physicalBlockId) {
        auto block = create_block_(prevBlock, blockSize);
        block->setBlockId(physicalBlockId);
        block->appendTokenIds(tokenIds);
        return block;
    }

    // In this minimal version, freeBlock is a no-op
    void freeBlock(const std::shared_ptr<Block>& block) {
        // No pooling logic implemented
    }

private:
    int blockSize_;
    BlockFactory create_block_;
    int poolSize_;
};

// Minimal BlockList implementation for fast-access to physical block ids
class BlockList {
public:
    BlockList() {}

    // Update the list with new blocks
    void update(const std::vector<std::shared_ptr<Block>>& blocks) {
        blocks_ = blocks;
        blockIds_.clear();
        for (auto& block : blocks_) {
            blockIds_.push_back(block->getBlockId());
        }
    }

    // Append a new block to the list
    void append(const std::shared_ptr<Block>& block) {
        blocks_.push_back(block);
        blockIds_.push_back(block->getBlockId());
    }

    size_t size() const {
        return blocks_.size();
    }

    std::shared_ptr<Block>& operator[](size_t index) {
        return blocks_[index];
    }

    const std::vector<std::shared_ptr<Block>>& list() const {
        return blocks_;
    }

    const std::vector<int>& ids() const {
        return blockIds_;
    }

    void reset() {
        blocks_.clear();
        blockIds_.clear();
    }

private:
    std::vector<std::shared_ptr<Block>> blocks_;
    std::vector<int> blockIds_;
};

// CacheMetricData: tracks cache metrics for blocks
struct CacheMetricData {
    int num_completed_blocks = 0;
    float completed_block_cache_hit_rate = 0.0f;
    int num_incompleted_block_queries = 0;
    int num_incompleted_block_hit = 0;
    int blockSize = 1000;

    // Process a query and update metrics
    void query(bool hit) {
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

    // Calculate the overall hit rate
    float get_hit_rate() const {
        float incomplete_ratio = static_cast<float>(num_incompleted_block_queries) / blockSize;
        float total_blocks = num_completed_blocks + incomplete_ratio;
        if (total_blocks == 0) return 0.0f;
        float completed_block_hit = (num_completed_blocks > 0) ? (completed_block_cache_hit_rate * num_completed_blocks) : 0.0f;
        float incompleted_block_hit = (num_incompleted_block_queries > 0) ? ((static_cast<float>(num_incompleted_block_hit) / num_incompleted_block_queries) * incomplete_ratio) : 0.0f;
        return (completed_block_hit + incompleted_block_hit) / total_blocks;
    }
};

// Recursively retrieves all blocks from the beginning to the given last block
inline std::vector<std::shared_ptr<Block>> getAllBlocksRecursively(const std::shared_ptr<Block>& lastBlock) {
    std::vector<std::shared_ptr<Block>> allBlocks;
    std::function<void(const std::shared_ptr<Block>&)> recurse = [&](const std::shared_ptr<Block>& block) {
        if (block->getPrevBlock()) {
            recurse(block->getPrevBlock());
        }
        allBlocks.push_back(block);
    };
    recurse(lastBlock);
    return allBlocks;
}

#endif // PAGED_CORE_BLOCK_COMMON_H 