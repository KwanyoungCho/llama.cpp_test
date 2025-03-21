#ifndef NAIVE_BLOCK_H
#define NAIVE_BLOCK_H

#include "interfaces.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <deque>
#include <set>

// 전방 선언: BlockAllocator
class BlockAllocator;

class NaiveBlock : public Block, public std::enable_shared_from_this<NaiveBlock> {
public:
    // 생성자 및 소멸자
    NaiveBlock(std::shared_ptr<Block> prevBlock, int blockSize, BlockAllocator* allocator,
               const std::vector<int>& initialTokenIds = std::vector<int>(),
               int blockId = -1,
               std::shared_ptr<Block> cowTarget = nullptr);
    ~NaiveBlock() override;

    // Block 인터페이스 구현
    void appendTokenIds(const std::vector<int>& tokenIds) override;
    int getBlockId() const override;
    void setBlockId(int id) override;
    std::vector<int>& getTokenIds() override;
    int numEmptySlots() const override;
    bool isFull() const override;
    std::shared_ptr<Block> getPrevBlock() const override;

    // 추가 기능: computed, lastAccessed, extra_hash, content_hash
    bool computed() const;
    void setComputed(bool value);
    double lastAccessed() const;
    void setLastAccessed(double timestamp);
    int getExtraHash() const;
    int getContentHash() const;

private:
    // 내부 헬퍼 함수
    void _appendTokenIdsNoCOW(const std::vector<int>& tokenIds);

    // 멤버 변수
    int _blockId;
    std::vector<int> _tokenIds;
    std::shared_ptr<Block> _prevBlock;
    int _blockSize;
    BlockAllocator* _allocator;
    std::shared_ptr<Block> _cowTarget; // copy-on-write 대상
};

class NaiveBlockAllocator : public BlockAllocator {
public:
    // 생성자
    NaiveBlockAllocator(Block::Factory createBlock, int numBlocks, int blockSize, const std::vector<int>& blockIds = std::vector<int>());

    // 블록 할당 관련 함수들
    std::shared_ptr<Block> allocateImmutableBlock(std::shared_ptr<Block> prevBlock, const std::vector<int>& tokenIds,
                                                    int extraHash = 0, Device device = Device::GPU);
    std::vector<std::shared_ptr<Block>> allocateImmutableBlocks(std::shared_ptr<Block> prevBlock, const std::vector<std::vector<int>>& blockTokenIds,
                                                                  int extraHash = 0, Device device = Device::GPU);
    std::shared_ptr<Block> allocateMutableBlock(std::shared_ptr<Block> prevBlock, int extraHash = 0, Device device = Device::GPU);
    int cow_block_if_not_appendable(std::shared_ptr<Block> block);
    void free(std::shared_ptr<Block> block);
    void freeBlockId(int blockId);
    std::vector<std::shared_ptr<Block>> fork(std::shared_ptr<Block> lastBlock);
    int getNumFreeBlocks() const;
    int getNumTotalBlocks() const;
    int getPhysicalBlockId(int absoluteId) const;
    std::vector<std::pair<int,int>> clearCopyOnWrites();
    void markBlocksAsAccessed(const std::vector<int>& blockIds, double now);
    void markBlocksAsComputed(const std::vector<int>& blockIds);
    std::vector<int> getCommonComputedBlockIds(const std::vector<std::vector<int>>& computedSeqBlockIds);
    int promoteToImmutableBlock(std::shared_ptr<Block> block);
    int getNumFullBlocksTouched(const std::vector<std::shared_ptr<Block>>& blocks);
    void swapOut(const std::vector<std::shared_ptr<Block>>& blocks);
    void swapIn(const std::vector<std::shared_ptr<Block>>& blocks);
    double getPrefixCacheHitRate() const;
    bool resetPrefixCache();
    std::vector<int> findCachedBlocksPrefix(const std::vector<int>& blockHashes);
    int blockSize() const;

    Block::Factory createBlock;

private:
    int _allocateBlockId();
    void _freeBlockId(std::shared_ptr<Block> block);
    void _freeBlockId(int blockId);

    int _blockSize;
    std::deque<int> _freeBlockIndices;
    std::set<int> _allBlockIndices;
};

#endif // NAIVE_BLOCK_H