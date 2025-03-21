#ifndef PREFIX_CACHING_BLOCK_H
#define PREFIX_CACHING_BLOCK_H

#include "interfaces.h"
#include <vector>
#include <memory>
#include <cassert>

// PrefixCachingBlock implements a block that supports prefix caching.
// In addition to the basic Block functionalities, it maintains a computed flag,
// a last accessed timestamp, and an extra hash value.

class PrefixCachingBlock : public Block {
public:
    // Constructor: accepts a previous block and the block size.
    PrefixCachingBlock(std::shared_ptr<Block> prevBlock, int blockSize)
        : prevBlock_(prevBlock), blockSize_(blockSize), blockId_(-1),
          computed_(false), lastAccessed_(0.0), extraHash_(0) {}

    ~PrefixCachingBlock() override = default;

    // Appends token IDs to the block. Ensures that the tokens to be added do
    // not exceed the available empty slots.
    void appendTokenIds(const std::vector<int>& tokenIds) override {
        assert(tokenIds.size() <= static_cast<size_t>(numEmptySlots()));
        tokenIds_.insert(tokenIds_.end(), tokenIds.begin(), tokenIds.end());
    }

    // Getter for block ID
    int getBlockId() const override {
        return blockId_;
    }

    // Setter for block ID
    void setBlockId(int id) override {
        blockId_ = id;
    }

    // Returns the token IDs stored in the block
    std::vector<int>& getTokenIds() override {
        return tokenIds_;
    }

    // Returns the number of empty slots remaining in the block
    int numEmptySlots() const override {
        return blockSize_ - static_cast<int>(tokenIds_.size());
    }

    // Checks if the block is full
    bool isFull() const override {
        return numEmptySlots() == 0;
    }

    // Returns the previous block in the sequence
    std::shared_ptr<Block> getPrevBlock() const override {
        return prevBlock_;
    }

    // Additional functionalities for prefix caching:

    // Gets the computed state
    bool getComputed() const { return computed_; }
    
    // Sets the computed state
    void setComputed(bool comp) { computed_ = comp; }

    // Gets the last accessed timestamp
    double getLastAccessed() const { return lastAccessed_; }
    
    // Sets the last accessed timestamp
    void setLastAccessed(double ts) { lastAccessed_ = ts; }

    // Gets the extra hash value
    int getExtraHash() const { return extraHash_; }
    
    // Sets the extra hash value
    void setExtraHash(int hash) { extraHash_ = hash; }

private:
    int blockId_;
    std::vector<int> tokenIds_;
    std::shared_ptr<Block> prevBlock_;
    int blockSize_;

    // Additional fields for prefix caching
    bool computed_;
    double lastAccessed_;
    int extraHash_;
};

#endif // PREFIX_CACHING_BLOCK_H 