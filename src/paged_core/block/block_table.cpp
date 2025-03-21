#include "block_table.h"
#include <cassert>
#include <cmath>
#include <algorithm>
// #include <optional>

// 기본 생성자: 기본 매핑 기능만 사용할 경우
BlockTable::BlockTable()
    : _block_size(0), _allocator(nullptr), _num_full_slots(0), _max_block_sliding_window(-1) {
    // _block_list는 기본적으로 빈 리스트로 초기화됨
}

// 확장 기능 생성자
BlockTable::BlockTable(int block_size, CPUGPUBlockAllocator* allocator,
                       const std::vector<std::shared_ptr<Block>>& init_blocks,
                       int max_block_sliding_window)
    : _block_size(block_size),
      _allocator(allocator),
      _max_block_sliding_window(max_block_sliding_window) {
    _block_list.update(init_blocks);
    _num_full_slots = get_num_token_ids();
}

BlockTable::~BlockTable() {
    // 필요한 경우 소멸자에서 자원 해제 로직 추가
}

// [기본 매핑 기능]

void BlockTable::addBlock(int blockId, std::shared_ptr<Block> block) {
    table_[blockId] = block;
}

std::shared_ptr<Block> BlockTable::getBlock(int blockId) const {
    auto it = table_.find(blockId);
    if (it != table_.end()) {
        return it->second;
    }
    return nullptr;
}

void BlockTable::removeBlock(int blockId) {
    table_.erase(blockId);
}

void BlockTable::clear() {
    table_.clear();
}

void BlockTable::updateBlock(int oldBlockId, int newBlockId, std::shared_ptr<Block> block) {
    table_.erase(oldBlockId);
    table_[newBlockId] = block;
}

// [확장 기능]

// Allocates blocks for the given token sequence.
void BlockTable::allocate(const std::vector<int>& token_ids) {
    // 할당되지 않은 상태여야 함
    assert(_block_list.list().empty() && "Blocks already allocated");
    assert(!token_ids.empty() && "token_ids should not be empty");

    // token_ids를 _block_size 크기로 분할
    std::vector< std::vector<int> > full_chunks;
    std::vector<int> tail_chunk;
    int n = token_ids.size();
    int fullChunksCount = n / _block_size;
    int remainder = n % _block_size;
    for (int i = 0; i < fullChunksCount; i++) {
        std::vector<int> chunk(token_ids.begin() + i * _block_size, token_ids.begin() + (i + 1) * _block_size);
        full_chunks.push_back(chunk);
    }
    if (remainder > 0) {
        tail_chunk = std::vector<int>(token_ids.begin() + fullChunksCount * _block_size, token_ids.end());
    }

    std::vector<std::shared_ptr<Block>> new_blocks;
    std::shared_ptr<Block> prev_block = nullptr;
    
    // immutable blocks 할당 (완전한 블록 단위)
    if (!full_chunks.empty()) {
        // extra_hash 제거: extra_hash 대신 호출 인자에서 제거
        std::vector<std::shared_ptr<Block>> immutable_blocks = _allocator->allocate_immutable_blocks(prev_block, full_chunks, Device::CPU);
        new_blocks.insert(new_blocks.end(), immutable_blocks.begin(), immutable_blocks.end());
        if (!immutable_blocks.empty()) {
            prev_block = immutable_blocks.back();
        }
    }
    
    // tail이 존재하면 mutable block 할당 후 토큰 추가
    if (!tail_chunk.empty()) {
        std::shared_ptr<Block> mutable_block = _allocator->allocate_mutable_block(prev_block, Device::CPU);
        mutable_block->appendTokenIds(tail_chunk);
        new_blocks.push_back(mutable_block);
    }

    // 새로 할당한 블록들로 업데이트
    update(new_blocks);
    _num_full_slots = token_ids.size();
}

// Updates the table with newly allocated blocks.
void BlockTable::update(const std::vector<std::shared_ptr<Block>>& blocks) {
    // _block_list가 update() 인터페이스를 제공한다고 가정
    _block_list.update(blocks);
}

// Appends token IDs to existing blocks.
void BlockTable::append_token_ids(const std::vector<int>& token_ids,
                                  int num_lookahead_slots,
                                  int num_computed_slots) {
    // 블록이 이미 할당되어 있어야 함
    assert(!_block_list.list().empty() && "No blocks allocated");

    // sliding window 기능 적용 (max_block_sliding_window가 사용중일 경우)
    if (_max_block_sliding_window != -1) {
        std::shared_ptr<Block> null_block = _allocator->allocate_or_get_null_block();
        assert(num_computed_slots != -1 && "num_computed_slots must be provided for sliding window");
        int end_block_idx = (num_computed_slots / _block_size) - _max_block_sliding_window;
        auto& blocks_list = const_cast<std::vector<std::shared_ptr<Block>>&>(_block_list.list());
        for (int idx = 0; idx < end_block_idx && idx < static_cast<int>(blocks_list.size()); idx++) {
            std::shared_ptr<Block> b = blocks_list[idx];
            if (b != null_block) {
                _allocator->free(b.get());
                blocks_list[idx] = null_block;
            }
        }
    }

    // 새로운 토큰들을 추가하기 위한 빈 슬롯 확보 (extra_hash 제거됨)
    ensure_num_empty_slots(token_ids.size() + num_lookahead_slots);

    int first_block_idx = _num_full_slots / _block_size;
    std::vector< std::vector<int> > token_blocks = chunk_token_blocks_for_append(token_ids);

    auto& blocks_list = const_cast<std::vector<std::shared_ptr<Block>>&>(_block_list.list());
    // 각 블록에 토큰 블록을 추가
    for (size_t i = 0; i < token_blocks.size(); i++) {
        int idx = first_block_idx + i;
        assert(idx < static_cast<int>(blocks_list.size()));
        blocks_list[idx]->appendTokenIds(token_blocks[i]);
    }
    _num_full_slots += token_ids.size();
}

// Ensures that there are at least 'num_empty_slots' free slots in the block table.
void BlockTable::ensure_num_empty_slots(int num_empty_slots) {
    assert(!_block_list.list().empty() && "No blocks allocated");
    int current_empty_slots = _block_list.list().size() * _block_size - _num_full_slots;
    if (current_empty_slots >= num_empty_slots)
        return;
    int slots_to_allocate = num_empty_slots - current_empty_slots;
    int blocks_to_allocate = cdiv(slots_to_allocate, _block_size);

    for (int i = 0; i < blocks_to_allocate; i++) {
        auto& blocks_list = const_cast<std::vector<std::shared_ptr<Block>>&>(_block_list.list());
        std::shared_ptr<Block> last_block = blocks_list.back();
        std::shared_ptr<Block> new_block = _allocator->allocate_mutable_block(last_block, Device::CPU);
        blocks_list.push_back(new_block);
    }
}

// Frees all blocks managed by this table.
void BlockTable::free_all() {
    std::vector<std::shared_ptr<Block>> blist = _block_list.list();
    for (auto& block : blist) {
        _allocator->free(block.get());
    }
    _block_list.reset();
}

// Returns a list of physical block IDs.
std::vector<int> BlockTable::physical_block_ids() const {
    return _block_list.ids();
}

// Given a full sequence of token IDs, returns the postfix that hasn't been appended.
std::vector<int> BlockTable::get_unseen_token_ids(const std::vector<int>& sequence_token_ids) const {
    if (static_cast<int>(sequence_token_ids.size()) <= _num_full_slots)
        return std::vector<int>();
    return std::vector<int>(sequence_token_ids.begin() + _num_full_slots, sequence_token_ids.end());
}

// Total number of tokens stored in the BlockTable.
int BlockTable::num_full_slots() const {
    return _num_full_slots;
}

// Determines how many blocks will be affected by appending token_ids (plus optional lookahead slots).
int BlockTable::get_num_blocks_touched_by_append_slots(const std::vector<int>& token_ids,
                                                         int num_lookahead_slots) const {
    int num_token_ids = token_ids.size() + num_lookahead_slots;
    int remainder = _num_full_slots % _block_size;
    int first_chunk_size = _block_size - remainder;
    if (num_token_ids <= first_chunk_size)
        return 1;
    int blocks = 1 + static_cast<int>(std::ceil((num_token_ids - first_chunk_size) / static_cast<double>(_block_size)));
    return blocks;
}

// Returns the underlying list of blocks.
std::vector<std::shared_ptr<Block>> BlockTable::blocks() const {
    return _block_list.list();
}

// --- Private Helper Methods ---

// Splits the token_ids into block-sized chunks so they can be appended to blocks.
std::vector<std::vector<int>> BlockTable::chunk_token_blocks_for_append(const std::vector<int>& token_ids) const {
    std::vector<std::vector<int>> chunks;
    if (token_ids.empty())
        return chunks;
    int remainder = _num_full_slots % _block_size;
    int first_chunk_size = _block_size - remainder;
    int n = token_ids.size();
    int end_first = std::min(first_chunk_size, n);
    chunks.push_back(std::vector<int>(token_ids.begin(), token_ids.begin() + end_first));
    int idx = end_first;
    while (idx < n) {
        int chunk_size = std::min(_block_size, n - idx);
        chunks.push_back(std::vector<int>(token_ids.begin() + idx, token_ids.begin() + idx + chunk_size));
        idx += chunk_size;
    }
    return chunks;
}

// Calculates the total number of token IDs stored in all blocks.
int BlockTable::get_num_token_ids() const {
    int total = 0;
    std::vector<std::shared_ptr<Block>> blist = _block_list.list();
    for (auto& block : blist) {
        // Assumes Block provides num_tokens() method returning the number of tokens stored
        total += block->numTokensTotal();
    }
    return total;
} 