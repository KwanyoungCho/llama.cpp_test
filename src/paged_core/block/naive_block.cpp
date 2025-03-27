#include "naive_block.h"
#include <algorithm>
#include <stdexcept>
#include <deque>
#include <set>
#include <unordered_set>
#include <numeric>

// ------------------ NaiveBlock 구현 ------------------

NaiveBlock::NaiveBlock(std::shared_ptr<Block> prev_block,
                       const std::vector<int>& token_ids,
                       int block_size,
                       BlockAllocator* allocator,
                       int block_id,
                       std::shared_ptr<Block> cow_target)
    : _token_ids(),
      _prev_block(prev_block),
      _block_size(block_size),
      _allocator(allocator),
      _block_id(block_id),
      _cow_target(cow_target ? cow_target : nullptr) {
    _append_token_ids_no_cow(token_ids);
}

NaiveBlock::~NaiveBlock() = default;

void NaiveBlock::append_token_ids(const std::vector<int>& token_ids) {
    _append_token_ids_no_cow(token_ids);
    if (_block_id != -1) {
        _block_id = _allocator->cow_block_if_not_appendable(*_cow_target);
    }
}

void NaiveBlock::_append_token_ids_no_cow(const std::vector<int>& token_ids) {
    if (token_ids.empty()) return;
    if (token_ids.size() > static_cast<size_t>(num_empty_slots())) {
        throw std::runtime_error("Not enough empty slots in block");
    }
    _token_ids.insert(_token_ids.end(), token_ids.begin(), token_ids.end());
}

const std::vector<int>& NaiveBlock::token_ids() const {
    return _token_ids;
}

std::vector<int>& NaiveBlock::token_ids() {
    return _token_ids;
}

bool NaiveBlock::is_full() const {
    return num_empty_slots() == 0;
}

int NaiveBlock::num_empty_slots() const {
    return _block_size - static_cast<int>(_token_ids.size());
}

std::shared_ptr<Block> NaiveBlock::prev_block() const {
    return _prev_block;
}

void NaiveBlock::set_prev_block(std::shared_ptr<Block> prev_block) {
    _prev_block = prev_block;
}

int NaiveBlock::block_id() const {
    return _block_id;
}

void NaiveBlock::set_block_id(int block_id) {
    _block_id = block_id;
}

bool NaiveBlock::computed() const {
    throw std::runtime_error("computed not implemented");
}

void NaiveBlock::set_computed(bool value) {
    throw std::runtime_error("set_computed not implemented");
}

double NaiveBlock::last_accessed() const {
    throw std::runtime_error("last_accessed not implemented");
}

void NaiveBlock::set_last_accessed(double timestamp) {
    throw std::runtime_error("set_last_accessed not implemented");
}

int NaiveBlock::block_size() const {
    return _block_size;
}

int NaiveBlock::num_tokens_total() const {
    throw std::runtime_error("num_tokens_total is not used for naive block");
}

int NaiveBlock::content_hash() const {
    return 0;
}

int NaiveBlock::pool_id() const {
    return _pool_id;
}

void NaiveBlock::set_pool_id(int pool_id) {
    _pool_id = pool_id;
}

// ------------------ NaiveBlockAllocator 구현 ------------------

NaiveBlockAllocator::NaiveBlockAllocator(std::shared_ptr<Block::Factory> create_block,
                                       int num_blocks,
                                       int block_size,
                                       const std::vector<int>& block_ids,
                                       std::shared_ptr<BlockPool> block_pool)
    : _create_block(create_block),
      _block_size(block_size) {
    
    std::vector<int> indices;
    if (block_ids.empty()) {
        indices.resize(num_blocks);
        std::iota(indices.begin(), indices.end(), 0);
    } else {
        indices = block_ids;
    }
    
    _free_block_indices = std::deque<int>(indices.begin(), indices.end());
    _all_block_indices = std::unordered_set<int>(indices.begin(), indices.end());
    
    _ref_counter = std::make_unique<RefCounter>(indices);
    _cow_tracker = std::make_unique<CopyOnWriteTracker>(*_ref_counter);

    if (block_pool) {
        _block_pool = std::unique_ptr<BlockPool>(block_pool.get());
    } else {
        const int extra_factor = 4;
        _block_pool = std::make_unique<BlockPool>(_block_size, 
            [this](std::shared_ptr<Block> prev_block, int block_size) {
                return std::make_shared<NaiveBlock>(prev_block, std::vector<int>(), 
                                                  block_size, this, -1);
            }, 
            num_blocks * extra_factor);
    }
}

std::shared_ptr<Block> NaiveBlockAllocator::allocate_mutable_block(std::shared_ptr<Block> prev_block) {
    int block_id = _allocate_block_id();
    auto block = _block_pool->init_block(prev_block, std::vector<int>(), _block_size, block_id);
    return block;
}

std::shared_ptr<Block> NaiveBlockAllocator::allocate_immutable_block(
    std::shared_ptr<Block> prev_block, const std::vector<int>& token_ids) {
    auto block = allocate_mutable_block(prev_block);
    block->append_token_ids(token_ids);
    return block;
}

std::vector<std::shared_ptr<Block>> NaiveBlockAllocator::allocate_immutable_blocks(
    std::shared_ptr<Block> prev_block,
    const std::vector<std::vector<int>>& block_token_ids) {
    int num_blocks = static_cast<int>(block_token_ids.size());
    
    std::vector<int> block_ids;
    block_ids.reserve(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        block_ids.push_back(_allocate_block_id());
    }

    std::vector<std::shared_ptr<Block>> blocks;
    blocks.reserve(num_blocks);
    
    for (int i = 0; i < num_blocks; ++i) {
        auto block = _block_pool->init_block(prev_block, block_token_ids[i],
                                           _block_size, block_ids[i]);
        blocks.push_back(block);
        prev_block = block;
    }
    
    return blocks;
}

void NaiveBlockAllocator::free(std::shared_ptr<Block> block, bool keep_block_object) {
    if (!block) return;
    _free_block_id(block);
    
    if (!keep_block_object) {
        _block_pool->free_block(block);
    }
}

void NaiveBlockAllocator::free_block_id(int block_id) {
    _free_block_id(block_id);
}

std::vector<std::shared_ptr<Block>> NaiveBlockAllocator::fork(std::shared_ptr<Block> last_block) {
    auto source_blocks = get_all_blocks_recursively(last_block);
    std::vector<std::shared_ptr<Block>> forked_blocks;
    forked_blocks.reserve(source_blocks.size());
    
    std::shared_ptr<Block> prev_block = nullptr;
    for (const auto& block : source_blocks) {
        if (!block->block_id()) {
            throw std::runtime_error("can't fork block with no block_id");
        }
        
        _ref_counter->incr(block->block_id());
        if (_ref_counter->get(block->block_id()) == 1) {
            throw std::runtime_error("can't fork free'd block");
        }
        
        auto forked_block = _block_pool->init_block(prev_block, block->token_ids(),
                                                  _block_size, block->block_id());
        forked_blocks.push_back(forked_block);
        prev_block = forked_block;
    }
    
    return forked_blocks;
}

void NaiveBlockAllocator::swap_out(const std::vector<std::shared_ptr<Block>>& blocks) {
    for (const auto& block : blocks) {
        _free_block_id(block);
    }
}

void NaiveBlockAllocator::swap_in(const std::vector<std::shared_ptr<Block>>& blocks) {
    for (const auto& block : blocks) {
        if (!block) continue;
        
        std::shared_ptr<Block> tmp_block;
        if (block->is_full()) {
            tmp_block = allocate_immutable_block(block->prev_block(), block->token_ids());
        } else {
            tmp_block = allocate_mutable_block(block->prev_block());
            tmp_block->append_token_ids(block->token_ids());
        }
        
        int block_id = tmp_block->block_id();
        tmp_block->set_block_id(-1);
        _block_pool->free_block(tmp_block);
        
        block->set_block_id(block_id);
    }
}

void NaiveBlockAllocator::mark_blocks_as_accessed(const std::vector<int>& block_ids, double now) {
    // Naive implementation doesn't track access times
}

void NaiveBlockAllocator::mark_blocks_as_computed(const std::vector<int>& block_ids) {
    // Naive implementation doesn't track computed status
}

std::vector<int> NaiveBlockAllocator::get_common_computed_block_ids(
    const std::vector<std::vector<int>>& computed_seq_block_ids) {
    return std::vector<int>();  // Naive implementation doesn't support prefix caching
}

int NaiveBlockAllocator::cow_block_if_not_appendable(const Block& block) {
    int src_block_id = block.block_id();
    if (src_block_id == -1) {
        throw std::runtime_error("Invalid block ID");
    }
    
    if (_cow_tracker->is_appendable(std::make_shared<NaiveBlock>(
            block.prev_block(), block.token_ids(), block.block_size(), this, src_block_id))) {
        return src_block_id;
    }
    
    _free_block_id(src_block_id);
    int trg_block_id = _allocate_block_id();
    _cow_tracker->record_cow(src_block_id, trg_block_id);
    return trg_block_id;
}

int NaiveBlockAllocator::promote_to_immutable_block(const Block& block) {
    throw std::runtime_error("Promotion not supported in naive implementation");
}

int NaiveBlockAllocator::get_num_full_blocks_touched(
    const std::vector<std::shared_ptr<Block>>& blocks) {
    std::unordered_set<int> old_block_set;
    for (const auto& block : blocks) {
        if (block && block->is_full()) {
            old_block_set.insert(block->block_id());
        }
    }
    return static_cast<int>(old_block_set.size());
}

int NaiveBlockAllocator::get_physical_block_id(int absolute_id) {
    std::vector<int> sorted(_all_block_indices.begin(), _all_block_indices.end());
    std::sort(sorted.begin(), sorted.end());
    auto it = std::find(sorted.begin(), sorted.end(), absolute_id);
    if (it == sorted.end()) {
        throw std::runtime_error("Invalid absolute block ID");
    }
    return std::distance(sorted.begin(), it);
}

const std::unordered_set<int>& NaiveBlockAllocator::all_block_ids() const {
    return _all_block_indices;
}

int NaiveBlockAllocator::get_num_total_blocks() const {
    return static_cast<int>(_all_block_indices.size());
}

int NaiveBlockAllocator::get_num_free_blocks() const {
    return static_cast<int>(_free_block_indices.size());
}

int NaiveBlockAllocator::_allocate_block_id() {
    if (_free_block_indices.empty()) {
        throw std::runtime_error("No free blocks available");
    }
    int id = _free_block_indices.front();
    _free_block_indices.pop_front();
    _ref_counter->incr(id);
    return id;
}

void NaiveBlockAllocator::_free_block_id(std::shared_ptr<Block> block) {
    if (block) {
        int block_id = block->block_id();
        block->set_block_id(-1);
        _free_block_id(block_id);
    }
}

void NaiveBlockAllocator::_free_block_id(int block_id) {
    if (block_id != -1) {
        int refcount = _ref_counter->decr(block_id);
        if (refcount == 0) {
            _free_block_indices.push_front(block_id);
        }
    }
}

std::vector<std::pair<int, int>> NaiveBlockAllocator::clear_copy_on_writes() {
    return _cow_tracker->clear_cows();
}