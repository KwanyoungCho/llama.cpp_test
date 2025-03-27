#include "common.h"
#include <algorithm>
#include <functional>
#include <stdexcept>

// ------------------- RefCounter -------------------

template<typename Iterable>
RefCounter::RefCounter(const Iterable& all_block_indices) {
    std::unordered_set<BlockId> deduped(all_block_indices.begin(), all_block_indices.end());
    for (BlockId id : deduped) {
        refcounts_[id] = 0;
    }
}

// 템플릿 함수의 구체화를 위한 명시적 인스턴스화
template RefCounter::RefCounter(const std::deque<BlockId>&);

RefCount RefCounter::incr(BlockId block_id) {
    auto it = refcounts_.find(block_id);
    if (it == refcounts_.end()) {
        throw std::runtime_error("BlockId not found in RefCounter::incr");
    }
    int pre_incr_refcount = it->second;
    if (pre_incr_refcount < 0) {
        throw std::runtime_error("Negative refcount in RefCounter::incr");
    }
    
    int post_incr_refcount = pre_incr_refcount + 1;
    it->second = post_incr_refcount;
    return post_incr_refcount;
}

RefCount RefCounter::decr(BlockId block_id) {
    auto it = refcounts_.find(block_id);
    if (it == refcounts_.end()) {
        throw std::runtime_error("BlockId not found in RefCounter::decr");
    }
    int refcount = it->second;
    if (refcount <= 0) {
        throw std::runtime_error("Refcount must be greater than 0 in RefCounter::decr");
    }
    
    refcount -= 1;
    it->second = refcount;
    return refcount;
}

RefCount RefCounter::get(BlockId block_id) const {
    auto it = refcounts_.find(block_id);
    if (it == refcounts_.end()) {
        throw std::runtime_error("BlockId not found in RefCounter::get");
    }
    return it->second;
}

std::shared_ptr<ReadOnlyRefCounter> RefCounter::as_readonly() {
    return std::make_shared<ReadOnlyRefCounter>(*this);
}

// ------------------- ReadOnlyRefCounter -------------------

ReadOnlyRefCounter::ReadOnlyRefCounter(const RefCounter& refcounter) 
    : refcounter_(refcounter) {}

RefCount ReadOnlyRefCounter::get(BlockId block_id) const {
    return refcounter_.get(block_id);
}

RefCount ReadOnlyRefCounter::incr(BlockId) {
    throw std::runtime_error("Incr not allowed in ReadOnlyRefCounter");
}

RefCount ReadOnlyRefCounter::decr(BlockId) {
    throw std::runtime_error("Decr not allowed in ReadOnlyRefCounter");
}

// ------------------- CopyOnWriteTracker -------------------

CopyOnWriteTracker::CopyOnWriteTracker(const RefCounterProtocol& refcounter) 
    : refcounter_(refcounter) {}

bool CopyOnWriteTracker::is_appendable(const std::shared_ptr<Block>& block) const {
    int id = block->block_id();
    if (id < 0) {
        return true;
    }
    return refcounter_.get(id) <= 1;
}

void CopyOnWriteTracker::record_cow(BlockId src_block_id, BlockId trg_block_id) {
    if (src_block_id < 0 || trg_block_id < 0) {
        throw std::runtime_error("Invalid blockId in record_cow");
    }
    copy_on_writes_.emplace_back(src_block_id, trg_block_id);
}

std::vector<std::pair<BlockId, BlockId>> CopyOnWriteTracker::clear_cows() {
    auto cows = copy_on_writes_;
    copy_on_writes_.clear();
    return cows;
}

// ------------------- BlockPool -------------------

BlockPool::BlockPool(int block_size, BlockFactory create_block, BlockAllocator* allocator, int pool_size)
    : block_size_(block_size), create_block_(create_block), allocator_(allocator), pool_size_(pool_size) {
    if (pool_size_ < 0) {
        throw std::runtime_error("pool_size_ must be non-negative in BlockPool::BlockPool");
    }
    
    for (int i = 0; i < pool_size_; i++) {
        free_ids_.push_back(i);
        auto block = (*create_block_)(
            nullptr,                    // prev_block
            std::vector<int>(),        // token_ids
            block_size_,               // block_size
            allocator_,                // allocator
            -1,                        // block_id
            false                      // computed
        );
        if (auto pooled_block = std::dynamic_pointer_cast<PooledBlock>(block)) {
            pooled_block->set_pool_id(i);
        }
        pool_.push_back(block);
    }
}

void BlockPool::increase_pool() {
    int cur_pool_size = pool_size_;
    int new_pool_size = cur_pool_size * 2;
    pool_size_ = new_pool_size;

    for (int i = cur_pool_size; i < new_pool_size; i++) {
        free_ids_.push_back(i);
        auto block = (*create_block_)(
            nullptr,                    // prev_block
            std::vector<int>(),        // token_ids
            block_size_,               // block_size
            allocator_,                // allocator
            -1,                        // block_id
            false                      // computed
        );
        if (auto pooled_block = std::dynamic_pointer_cast<PooledBlock>(block)) {
            pooled_block->set_pool_id(i);
        }
        pool_.push_back(block);
    }
}

std::shared_ptr<Block> BlockPool::init_block(
    std::shared_ptr<Block> prev_block,
    const std::vector<int>& token_ids,
    int block_size,
    int physical_block_id) {
    
    if (free_ids_.empty()) {
        increase_pool();
        if (free_ids_.empty()) {
            throw std::runtime_error("free_ids_ is still empty in BlockPool::init_block after increasing pool");
        }
    }

    int pool_id = free_ids_.front();
    free_ids_.pop_front();

    auto block = (*create_block_)(
        prev_block,           // prev_block
        token_ids,           // token_ids
        block_size,          // block_size
        allocator_,          // allocator
        physical_block_id,   // block_id
        false                // computed
    );
    if (auto pooled_block = std::dynamic_pointer_cast<PooledBlock>(block)) {
        pooled_block->set_pool_id(pool_id);
    }
    pool_[pool_id] = block;
    return block;
}

void BlockPool::free_block(std::shared_ptr<Block> block) {
    if (auto pooled_block = std::dynamic_pointer_cast<PooledBlock>(block)) {
        int pool_id = pooled_block->pool_id();
        free_ids_.push_back(pool_id);
    } else {
        throw std::runtime_error("Block is not a PooledBlock in BlockPool::free_block");
    }
}

// ------------------- BlockList -------------------

BlockList::BlockList() {}

void BlockList::update(const std::vector<std::shared_ptr<Block>>& blocks) {
    blocks_ = blocks;
    block_ids_.clear();
    for (const auto& block : blocks_) {
        if (block->block_id() < 0) {
            throw std::runtime_error("BlockList::update found negative blockId");
        }
        block_ids_.push_back(block->block_id());
    }
}

void BlockList::append_token_ids(int block_index, const std::vector<int>& token_ids) {
    if (block_index < 0 || block_index >= static_cast<int>(blocks_.size())) {
        throw std::runtime_error("Invalid block index in BlockList::append_token_ids");
    }
    auto& block = blocks_[block_index];
    int prev_id = block->block_id();
    block->append_token_ids(token_ids);
    if (prev_id != block->block_id()) {
        block_ids_[block_index] = block->block_id();
    }
}

void BlockList::append(const std::shared_ptr<Block>& block) {
    blocks_.push_back(block);
    block_ids_.push_back(block->block_id());
}

size_t BlockList::size() const {
    return blocks_.size();
}

std::shared_ptr<Block>& BlockList::operator[](size_t index) {
    return blocks_[index];
}

const std::vector<std::shared_ptr<Block>>& BlockList::list() const {
    return blocks_;
}

const std::vector<int>& BlockList::ids() const {
    return block_ids_;
}

void BlockList::reset() {
    blocks_.clear();
    block_ids_.clear();
}

// ------------------- CacheMetricData -------------------

void CacheMetricData::query(bool hit) {
    num_incompleted_block_queries += 1;
    if (hit) {
        num_incompleted_block_hit += 1;
    }
    if (num_incompleted_block_queries == block_size) {
        float hit_rate = static_cast<float>(num_incompleted_block_hit) / num_incompleted_block_queries;
        completed_block_cache_hit_rate = (completed_block_cache_hit_rate * num_completed_blocks + hit_rate) / (num_completed_blocks + 1);
        num_incompleted_block_queries = 0;
        num_incompleted_block_hit = 0;
        num_completed_blocks += 1;
    }
}

float CacheMetricData::get_hit_rate() const {
    float incomplete_ratio = static_cast<float>(num_incompleted_block_queries) / block_size;
    float total_blocks = num_completed_blocks + incomplete_ratio;
    if (total_blocks == 0) return 0.0f;
    float completed_block_hit = (num_completed_blocks > 0) ? (completed_block_cache_hit_rate * num_completed_blocks) : 0.0f;
    float incompleted_block_hit = (num_incompleted_block_queries > 0)
        ? ((static_cast<float>(num_incompleted_block_hit) / num_incompleted_block_queries) * incomplete_ratio)
        : 0.0f;
    return (completed_block_hit + incompleted_block_hit) / total_blocks;
}

// ------------------- get_all_blocks_recursively -------------------

std::vector<std::shared_ptr<Block>> get_all_blocks_recursively(const std::shared_ptr<Block>& last_block) {
    std::vector<std::shared_ptr<Block>> all_blocks;
    std::function<void(const std::shared_ptr<Block>&)> recurse = [&](const std::shared_ptr<Block>& block) {
        if (block->prev_block()) {
            recurse(block->prev_block());
        }
        all_blocks.push_back(block);
    };
    recurse(last_block);
    return all_blocks;
}