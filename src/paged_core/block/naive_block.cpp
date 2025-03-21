#include "naive_block.h"
#include <algorithm>
#include <stdexcept>
#include <deque>
#include <set>
#include <unordered_set>

// ------------------ NaiveBlock 구현 ------------------

NaiveBlock::NaiveBlock(std::shared_ptr<Block> prev_block, int block_size, BlockAllocator* allocator,
                       const std::vector<int>& initial_token_ids, int block_id, std::shared_ptr<Block> cow_target)
    : _prevBlock(prev_block),
      _blockSize(block_size),
      _allocator(allocator),
      _blockId(block_id),
      _tokenIds(initial_token_ids),
      _cowTarget(cow_target ? cow_target : nullptr) {
}

NaiveBlock::~NaiveBlock() {
    // 필요한 정리 작업이 있다면 이곳에 추가합니다.
}

void NaiveBlock::appendTokenIds(const std::vector<int>& token_ids) {
    _appendTokenIdsNoCOW(token_ids);
    if (_blockId != -1) {
        if (!_cowTarget) {
            _cowTarget = shared_from_this();
        }
        _blockId = _allocator->cow_block_if_not_appendable(*_cowTarget);
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

int NaiveBlock::numTokensTotal() const {
    throw std::runtime_error("Not implemented: numTokensTotal is not used for naive block");
}

bool NaiveBlock::isComputed() const {
    throw std::runtime_error("Not implemented: isComputed getter");
}

void NaiveBlock::setComputed(bool value) {
    throw std::runtime_error("Not implemented: isComputed setter");
}

double NaiveBlock::getLastAccessed() const {
    throw std::runtime_error("Not implemented: lastAccessed getter");
}

void NaiveBlock::setLastAccessed(double timestamp) {
    throw std::runtime_error("Not implemented: lastAccessed setter");
}

int NaiveBlock::getContentHash() const {
    return 0;
}

void NaiveBlock::_appendTokenIdsNoCOW(const std::vector<int>& token_ids) {
    if (token_ids.empty()) return;
    if (token_ids.size() > static_cast<size_t>(numEmptySlots())) {
        throw std::runtime_error("Not enough empty slots");
    }
    _tokenIds.insert(_tokenIds.end(), token_ids.begin(), token_ids.end());
}

// ------------------ NaiveBlockAllocator 구현 ------------------

NaiveBlockAllocator::NaiveBlockAllocator(std::shared_ptr<Block::Factory> create_block, int num_blocks, int block_size, const std::vector<int>& block_ids)
    : createBlock(create_block), _blockSize(block_size) {
    if (block_ids.empty()) {
        for (int i = 0; i < num_blocks; i++) {
            _freeBlockIndices.push_back(i);
            _allBlockIndices.insert(i);
        }
    } else {
        for (int id : block_ids) {
            _freeBlockIndices.push_back(id);
            _allBlockIndices.insert(id);
        }
    }
}

std::shared_ptr<Block> NaiveBlockAllocator::allocate_mutable_block(std::shared_ptr<Block> prev_block) {
    int block_id = _allocateBlockId();
    // Factory 인터페이스를 사용하여 Block 객체를 생성 (빈 token_ids, computed=false)
    std::shared_ptr<Block> block = (*createBlock)(prev_block, std::vector<int>(), _blockSize, this, block_id, false);
    return block;
}

std::shared_ptr<Block> NaiveBlockAllocator::allocate_immutable_block(std::shared_ptr<Block> prev_block, const std::vector<int>& token_ids) {
    std::shared_ptr<Block> block = allocate_mutable_block(prev_block);
    block->appendTokenIds(token_ids);
    return block;
}

std::vector<std::shared_ptr<Block>> NaiveBlockAllocator::allocate_immutable_blocks(std::shared_ptr<Block> prev_block, const std::vector<std::vector<int>>& block_token_ids) {
    std::vector<std::shared_ptr<Block>> blocks;
    for (const auto& token_ids : block_token_ids) {
        std::shared_ptr<Block> block = allocate_immutable_block(prev_block, token_ids);
        blocks.push_back(block);
        prev_block = block;
    }
    return blocks;
}

void NaiveBlockAllocator::free(Block* block) {
    int id = block->getBlockId();
    _freeBlockId(id);
    block->setBlockId(-1);
}

std::vector<std::shared_ptr<Block>> NaiveBlockAllocator::fork(std::shared_ptr<Block> last_block) {
    return { last_block };
}

int NaiveBlockAllocator::get_num_free_blocks() const {
    return _freeBlockIndices.size();
}

int NaiveBlockAllocator::get_num_total_blocks() const {
    return _allBlockIndices.size();
}

int NaiveBlockAllocator::get_physical_block_id(int absolute_id) {
    std::vector<int> sorted(_allBlockIndices.begin(), _allBlockIndices.end());
    std::sort(sorted.begin(), sorted.end());
    auto it = std::find(sorted.begin(), sorted.end(), absolute_id);
    if (it != sorted.end())
        return std::distance(sorted.begin(), it);
    throw std::runtime_error("Block id not found");
}

std::vector<std::pair<int,int>> NaiveBlockAllocator::clear_copy_on_writes() {
    return {};
}

void NaiveBlockAllocator::mark_blocks_as_accessed(const std::vector<int>& block_ids, double now) {
    // No operation for naive allocator
}

void NaiveBlockAllocator::mark_blocks_as_computed(const std::vector<int>& block_ids) {
    // No operation for naive allocator
}

std::vector<int> NaiveBlockAllocator::get_common_computed_block_ids(const std::vector<std::vector<int>>& computed_seq_block_ids) {
    return {};
}

int NaiveBlockAllocator::cow_block_if_not_appendable(const Block& block) {
    return block.getBlockId();
}

int NaiveBlockAllocator::promote_to_immutable_block(const Block& block) {
    throw std::runtime_error("Promotion not supported for naive blocks");
}

int NaiveBlockAllocator::get_num_full_blocks_touched(const std::vector<std::shared_ptr<Block>>& blocks) {
    std::set<int> full_block_ids;
    for (auto& block : blocks) {
        if (block->isFull()) {
            full_block_ids.insert(block->getBlockId());
        }
    }
    return full_block_ids.size();
}

float NaiveBlockAllocator::get_prefix_cache_hit_rate() {
    return -1.0;
}

bool NaiveBlockAllocator::reset_prefix_cache() {
    return true;
}

std::vector<int> NaiveBlockAllocator::find_cached_blocks_prefix(const std::vector<int>& block_hashes) {
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

void NaiveBlockAllocator::_freeBlockId(int block_id) {
    _freeBlockIndices.push_back(block_id);
}

// 추가: swap_out, swap_in, all_block_ids 구현

void NaiveBlockAllocator::swap_out(const std::vector<std::shared_ptr<Block>>& blocks) {
    for (const auto& block : blocks) {
        free(block.get());
    }
}

void NaiveBlockAllocator::swap_in(const std::vector<std::shared_ptr<Block>>& blocks) {
    for (auto& block : blocks) {
        std::shared_ptr<Block> tmp_block;
        if (block->isFull()) {
            // 블록이 full이면 immutable 블록 할당 (이전에 연결된 prev_block, token_ids 사용)
            tmp_block = allocate_immutable_block(block->getPrevBlock(), block->getTokenIds());
        } else {
            // 그렇지 않으면 mutable 블록 할당 후 기존 토큰들을 추가
            tmp_block = allocate_mutable_block(block->getPrevBlock());
            tmp_block->appendTokenIds(block->getTokenIds());
        }
        int new_id = tmp_block->getBlockId();
        tmp_block->setBlockId(-1);
        // 임시 블록을 해제하여 블록 ID를 풀에 반환
        free(tmp_block.get());
        // 원래 블록에 새로운 ID 할당
        block->setBlockId(new_id);
    }
}

const std::unordered_set<int>& NaiveBlockAllocator::all_block_ids() const {
    return _allBlockIndices;
}