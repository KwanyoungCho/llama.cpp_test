#include "block_table.h"
#include "../utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
// #include <optional>

// 기본 생성자: 기본 매핑 기능만 사용할 경우
BlockTable::BlockTable()
    : _block_size(0), _allocator(nullptr), _num_full_slots(0), _max_block_sliding_window(-1) {
    // _blocks는 기본적으로 빈 리스트로 초기화됨
}

// 확장 기능 생성자
BlockTable::BlockTable(int block_size, DeviceAwareBlockAllocator* block_allocator,
                       const std::vector<std::shared_ptr<Block>>& _blocks,
                       int max_block_sliding_window)
    : _block_size(block_size),
      _allocator(block_allocator),
      _max_block_sliding_window(max_block_sliding_window),
      _blocks(_blocks) {  // BlockList 생성자에 직접 전달
    _num_full_slots = _get_num_token_ids();
}

BlockTable::~BlockTable() = default;

// 정적 메서드: 주어진 토큰 ID와 블록 크기를 고려하여 필요한 블록 수 계산
int BlockTable::get_num_required_blocks(const std::vector<int>& token_ids,
                                        int block_size,
                                        int num_lookahead_slots) {
    int total_slots = token_ids.size() + num_lookahead_slots;
    return (total_slots + block_size - 1) / block_size; // ceiling division
}

// Allocates blocks for the given token sequence.
void BlockTable::allocate(const std::vector<int>& token_ids, Device device) {
    if (_is_allocated()) {
        throw std::runtime_error("Blocks already allocated");
    }
    if (token_ids.empty()) {
        throw std::runtime_error("token_ids should not be empty");
    }

    auto blocks = _allocate_blocks_for_token_ids(nullptr, token_ids, device);
    update(blocks);
    _num_full_slots = token_ids.size();
}

// Updates the table with newly allocated blocks.
void BlockTable::update(const std::vector<std::shared_ptr<Block>>& blocks) {
    _blocks.update(blocks);
}

// Appends token IDs to existing blocks.
void BlockTable::append_token_ids(const std::vector<int>& token_ids,
                                  int num_lookahead_slots,
                                  int num_computed_slots) {
    if (!_is_allocated()) {
        throw std::runtime_error("no blocks have been allocated");
    }
    if (_blocks.size() == 0) {
        throw std::runtime_error("no blocks available");
    }

    // sliding window 기능 적용
    if (_max_block_sliding_window != -1) {
        std::shared_ptr<Block> null_block = _allocator->allocate_or_get_null_block();
        if (num_computed_slots == -1) {
            throw std::runtime_error("num_computed_slots must be provided for sliding window");
        }
        int end_block_idx = (num_computed_slots / _block_size) - _max_block_sliding_window;
        for (int idx = 0; idx < end_block_idx; idx++) {
            std::shared_ptr<Block> b = _blocks[idx];
            if (b != null_block) {
                _allocator->free(b.get());
                _blocks[idx] = null_block;
            }
        }
    }

    // 새로운 토큰들을 추가하기 위한 빈 슬롯 확보
    ensure_num_empty_slots(token_ids.size() + num_lookahead_slots);

    int first_block_idx = _num_full_slots / _block_size;
    std::vector<std::vector<int>> token_blocks = _chunk_token_blocks_for_append(token_ids);

    for (size_t i = 0; i < token_blocks.size(); i++) {
        _blocks.append_token_ids(first_block_idx + i, token_blocks[i]);
    }
    _num_full_slots += token_ids.size();
}

// Ensures that there are at least 'num_empty_slots' free slots in the block table.
void BlockTable::ensure_num_empty_slots(int num_empty_slots) {
    if (!_is_allocated()) {
        throw std::runtime_error("no blocks have been allocated");
    }
    Device device = Device::GPU;  // 현재는 GPU만 지원

    int current_empty_slots = _num_empty_slots();
    if (current_empty_slots >= num_empty_slots)
        return;

    int slots_to_allocate = num_empty_slots - current_empty_slots;
    int blocks_to_allocate = (slots_to_allocate + _block_size - 1) / _block_size;

    for (int i = 0; i < blocks_to_allocate; i++) {
        if (_blocks.size() == 0) {
            throw std::runtime_error("no blocks available");
        }
        std::shared_ptr<Block> last_block = _blocks[_blocks.size() - 1];
        std::shared_ptr<Block> new_block = _allocator->allocate_mutable_block(last_block, device);
        _blocks.append(new_block);
    }
}

// Create a fork of the BlockTable
std::shared_ptr<BlockTable> BlockTable::fork() {
    if (!_is_allocated()) {
        throw std::runtime_error("no blocks have been allocated");
    }
    if (_blocks.size() == 0) {
        throw std::runtime_error("no blocks available");
    }
    auto forked_blocks = _allocator->fork(_blocks[_blocks.size() - 1]);
    return std::make_shared<BlockTable>(
        _block_size,
        _allocator,
        forked_blocks,
        _max_block_sliding_window
    );
}

// Frees all blocks managed by this table.
void BlockTable::free() {
    for (auto& block : blocks()) {
        _allocator->free(block.get());
    }
    _blocks.reset();
}

// Returns a list of physical block IDs.
std::vector<int> BlockTable::physical_block_ids() const {
    return _blocks.ids();
}

// Given a full sequence of token IDs, returns the postfix that hasn't been appended.
std::vector<int> BlockTable::get_unseen_token_ids(const std::vector<int>& sequence_token_ids) const {
    if (static_cast<int>(sequence_token_ids.size()) <= _num_full_slots)
        return std::vector<int>();
    return std::vector<int>(sequence_token_ids.begin() + _num_full_slots, sequence_token_ids.end());
}

// Properties (getter 메서드)
// Returns the underlying list of blocks.
std::vector<std::shared_ptr<Block>> BlockTable::blocks() const {
    return _blocks.list();
}

// Check if blocks are allocated
bool BlockTable::_is_allocated() const {
    return _blocks.size() > 0;
}

// Get the number of empty slots
int BlockTable::_num_empty_slots() const {
    if (!_is_allocated()) {
        throw std::runtime_error("no blocks have been allocated");
    }
    return _blocks.size() * _block_size - _num_full_slots;
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
    return 1 + static_cast<int>(std::ceil((num_token_ids - first_chunk_size) / static_cast<double>(_block_size)));
}

// --- Private Helper Methods ---

// Splits the token_ids into block-sized chunks so they can be appended to blocks.
std::vector<std::vector<int>> BlockTable::_chunk_token_blocks_for_append(const std::vector<int>& token_ids) const {
    if (token_ids.empty())
        return std::vector<std::vector<int>>();

    int remainder = _num_full_slots % _block_size;
    int first_chunk_size = _block_size - remainder;
    std::vector<std::vector<int>> token_blocks;
    
    // 첫 번째 청크
    token_blocks.push_back(std::vector<int>(token_ids.begin(), 
        token_ids.begin() + std::min(first_chunk_size, static_cast<int>(token_ids.size()))));
    
    // 나머지 청크들 - chunk_list 사용
    if (token_ids.size() > first_chunk_size) {
        std::vector<int> remaining_tokens(token_ids.begin() + first_chunk_size, token_ids.end());
        for (const auto& chunk : paged_core::chunk_list<int>(remaining_tokens, _block_size)) {
            token_blocks.push_back(chunk);
        }
    }
    
    return token_blocks;
}

// Calculates the total number of token IDs stored in all blocks.
int BlockTable::_get_num_token_ids() const {
    int res = 0;
    for (const auto& block : blocks()) {
        res += block->token_ids().size();
    }
    return res;
}

std::vector<int> BlockTable::_get_all_token_ids() const {
    std::vector<int> token_ids;
    if (!_is_allocated())
        return token_ids;

    for (const auto& block : blocks()) {
        token_ids.insert(token_ids.end(), block->token_ids().begin(), block->token_ids().end());
    }
    return token_ids;
}

std::vector<std::shared_ptr<Block>> BlockTable::_allocate_blocks_for_token_ids(
    std::shared_ptr<Block> prev_block,
    const std::vector<int>& token_ids,
    Device device) {
    
    std::vector<std::shared_ptr<Block>> blocks;
    std::vector<std::vector<int>> block_token_ids;
    std::vector<int> tail_token_ids;

    for (const auto& chunk : paged_core::chunk_list<int>(token_ids, _block_size)) {
        if (chunk.size() == _block_size) {
            block_token_ids.push_back(chunk);
        } else {
            tail_token_ids = chunk;
        }
    }

    if (!block_token_ids.empty()) {
        auto immutable_blocks = _allocator->allocate_immutable_blocks(
            prev_block, block_token_ids, device);
        blocks.insert(blocks.end(), immutable_blocks.begin(), immutable_blocks.end());
        if (!immutable_blocks.empty()) {
            prev_block = immutable_blocks.back();
        }
    }

    if (!tail_token_ids.empty()) {
        auto block = _allocator->allocate_mutable_block(prev_block, device);
        block->append_token_ids(tail_token_ids);
        blocks.push_back(block);
    }

    return blocks;
} 