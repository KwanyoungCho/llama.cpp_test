#include "naive_block.h"
#include <algorithm>
#include <stdexcept>
#include <deque>
#include <set>

// NaiveBlock 구현

NaiveBlock::NaiveBlock(std::shared_ptr<Block> prevBlock, int blockSize, BlockAllocator* allocator,
                       const std::vector<int>& initialTokenIds, int blockId, std::shared_ptr<Block> cowTarget)
    : _prevBlock(prevBlock),
      _blockSize(blockSize),
      _allocator(allocator),
      _blockId(blockId),
      _tokenIds(initialTokenIds) {
    _cowTarget = cowTarget ? cowTarget : nullptr;
}

NaiveBlock::~NaiveBlock() {
    // 필요한 정리 작업이 있다면 여기에 추가
}

void NaiveBlock::appendTokenIds(const std::vector<int>& tokenIds) {
    _appendTokenIdsNoCOW(tokenIds);
    if (_blockId != -1) {
        if (!_cowTarget) {
            _cowTarget = shared_from_this();
        }
        // allocator의 cow_block_if_not_appendable 함수를 호출하여 새로운 block id를 반환받음
        _blockId = _allocator->cow_block_if_not_appendable(_cowTarget);
    }
}

int NaiveBlock::getBlockId() const {
    return _blockId;
}

void NaiveBlock::setBlockId(int id) {
    _blockId = id;
}

std::vector<int>& NaiveBlock::getTokenIds() {
    return _tokenIds;
}

int NaiveBlock::numEmptySlots() const {
    return _blockSize - static_cast<int>(_tokenIds.size());
}

bool NaiveBlock::isFull() const {
    return numEmptySlots() == 0;
}

std::shared_ptr<Block> NaiveBlock::getPrevBlock() const {
    return _prevBlock;
}

bool NaiveBlock::computed() const {
    throw std::runtime_error("Not implemented: computed getter");
}

void NaiveBlock::setComputed(bool value) {
    throw std::runtime_error("Not implemented: computed setter");
}

double NaiveBlock::lastAccessed() const {
    throw std::runtime_error("Not implemented: lastAccessed getter");
}

void NaiveBlock::setLastAccessed(double timestamp) {
    throw std::runtime_error("Not implemented: lastAccessed setter");
}

int NaiveBlock::getExtraHash() const {
    return 0;
}

int NaiveBlock::getContentHash() const {
    return 0;
}

void NaiveBlock::_appendTokenIdsNoCOW(const std::vector<int>& tokenIds) {
    if (tokenIds.empty()) return;
    if (tokenIds.size() > static_cast<size_t>(numEmptySlots())) {
        throw std::runtime_error("Not enough empty slots");
    }
    _tokenIds.insert(_tokenIds.end(), tokenIds.begin(), tokenIds.end());
}

// NaiveBlockAllocator 구현

NaiveBlockAllocator::NaiveBlockAllocator(Block::Factory createBlock, int numBlocks, int blockSize, const std::vector<int>& blockIds)
    : createBlock(createBlock), _blockSize(blockSize) {
    if (blockIds.empty()) {
        for (int i = 0; i < numBlocks; i++) {
            _freeBlockIndices.push_back(i);
            _allBlockIndices.insert(i);
        }
    } else {
        for (int id : blockIds) {
            _freeBlockIndices.push_back(id);
            _allBlockIndices.insert(id);
        }
    }
}

std::shared_ptr<Block> NaiveBlockAllocator::allocateImmutableBlock(std::shared_ptr<Block> prevBlock, const std::vector<int>& tokenIds,
                                                                     int extraHash, Device device) {
    auto block = allocateMutableBlock(prevBlock, extraHash, device);
    block->appendTokenIds(tokenIds);
    return block;
}

std::vector<std::shared_ptr<Block>> NaiveBlockAllocator::allocateImmutableBlocks(std::shared_ptr<Block> prevBlock,
                                                                                  const std::vector<std::vector<int>>& blockTokenIds,
                                                                                  int extraHash, Device device) {
    std::vector<std::shared_ptr<Block>> blocks;
    for (const auto& tokenIds : blockTokenIds) {
        auto block = allocateImmutableBlock(prevBlock, tokenIds, extraHash, device);
        blocks.push_back(block);
        prevBlock = block;
    }
    return blocks;
}

std::shared_ptr<Block> NaiveBlockAllocator::allocateMutableBlock(std::shared_ptr<Block> prevBlock, int extraHash, Device device) {
    int blockId = _allocateBlockId();
    std::shared_ptr<Block> block = createBlock(prevBlock, _blockSize, this, blockId);
    return block;
}

int NaiveBlockAllocator::cow_block_if_not_appendable(std::shared_ptr<Block> block) {
    // 간단 구현: 항상 원래의 block id 반환
    return block->getBlockId();
}

void NaiveBlockAllocator::free(std::shared_ptr<Block> block) {
    _freeBlockId(block);
}

void NaiveBlockAllocator::freeBlockId(int blockId) {
    _freeBlockId(blockId);
}

std::vector<std::shared_ptr<Block>> NaiveBlockAllocator::fork(std::shared_ptr<Block> lastBlock) {
    return { lastBlock };
}

int NaiveBlockAllocator::getNumFreeBlocks() const {
    return _freeBlockIndices.size();
}

int NaiveBlockAllocator::getNumTotalBlocks() const {
    return _allBlockIndices.size();
}

int NaiveBlockAllocator::getPhysicalBlockId(int absoluteId) const {
    std::vector<int> sorted(_allBlockIndices.begin(), _allBlockIndices.end());
    std::sort(sorted.begin(), sorted.end());
    auto it = std::find(sorted.begin(), sorted.end(), absoluteId);
    if (it != sorted.end())
        return std::distance(sorted.begin(), it);
    throw std::runtime_error("Block id not found");
}

std::vector<std::pair<int,int>> NaiveBlockAllocator::clearCopyOnWrites() {
    return {};
}

void NaiveBlockAllocator::markBlocksAsAccessed(const std::vector<int>& blockIds, double now) {
    // no operation for naive allocator
}

void NaiveBlockAllocator::markBlocksAsComputed(const std::vector<int>& blockIds) {
    // no operation for naive allocator
}

std::vector<int> NaiveBlockAllocator::getCommonComputedBlockIds(const std::vector<std::vector<int>>& computedSeqBlockIds) {
    return {};
}

int NaiveBlockAllocator::promoteToImmutableBlock(std::shared_ptr<Block> block) {
    throw std::runtime_error("Promotion not supported for naive blocks");
}

int NaiveBlockAllocator::getNumFullBlocksTouched(const std::vector<std::shared_ptr<Block>>& blocks) {
    std::set<int> fullBlockIds;
    for (auto& block : blocks) {
        if (block->isFull()) {
            fullBlockIds.insert(block->getBlockId());
        }
    }
    return fullBlockIds.size();
}

void NaiveBlockAllocator::swapOut(const std::vector<std::shared_ptr<Block>>& blocks) {
    for (auto& block : blocks) {
        _freeBlockId(block);
    }
}

void NaiveBlockAllocator::swapIn(const std::vector<std::shared_ptr<Block>>& blocks) {
    for (auto& block : blocks) {
        int newId = _allocateBlockId();
        block->setBlockId(newId);
    }
}

double NaiveBlockAllocator::getPrefixCacheHitRate() const {
    return -1.0;
}

bool NaiveBlockAllocator::resetPrefixCache() {
    return true;
}

std::vector<int> NaiveBlockAllocator::findCachedBlocksPrefix(const std::vector<int>& blockHashes) {
    return {};
}

int NaiveBlockAllocator::blockSize() const {
    return _blockSize;
}

int NaiveBlockAllocator::_allocateBlockId() {
    if (_freeBlockIndices.empty()) {
        throw std::runtime_error("No free blocks available");
    }
    int id = _freeBlockIndices.front();
    _freeBlockIndices.pop_front();
    return id;
}

void NaiveBlockAllocator::_freeBlockId(std::shared_ptr<Block> block) {
    int id = block->getBlockId();
    _freeBlockIndices.push_back(id);
}

void NaiveBlockAllocator::_freeBlockId(int blockId) {
    _freeBlockIndices.push_back(blockId);
}