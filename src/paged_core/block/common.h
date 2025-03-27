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
#include <unordered_set>

using BlockId = int;
using RefCount = int;

// RefCounterProtocol: 참조 카운터의 인터페이스
class RefCounterProtocol {
public:
    virtual ~RefCounterProtocol() = default;
    virtual RefCount incr(BlockId block_id) = 0;
    virtual RefCount decr(BlockId block_id) = 0;
    virtual RefCount get(BlockId block_id) const = 0;
};

// RefCounter: 블록 인덱스 집합에 대한 참조 카운트를 관리
class RefCounter : public RefCounterProtocol {
public:
    template<typename Iterable>
    RefCounter(const Iterable& all_block_indices);
    
    RefCount incr(BlockId block_id) override;
    RefCount decr(BlockId block_id) override;
    RefCount get(BlockId block_id) const override;
    
    std::shared_ptr<class ReadOnlyRefCounter> as_readonly();

private:
    std::unordered_map<BlockId, RefCount> refcounts_;
};

// ReadOnlyRefCounter: RefCounter의 읽기 전용 뷰
class ReadOnlyRefCounter : public RefCounterProtocol {
public:
    explicit ReadOnlyRefCounter(const RefCounter& refcounter);
    
    RefCount incr(BlockId block_id) override;
    RefCount decr(BlockId block_id) override;
    RefCount get(BlockId block_id) const override;

private:
    const RefCounter& refcounter_;
};

// CopyOnWriteTracker: 블록의 copy-on-write 작업을 추적하고 관리
class CopyOnWriteTracker {
public:
    explicit CopyOnWriteTracker(const RefCounterProtocol& refcounter);
    
    bool is_appendable(const std::shared_ptr<Block>& block) const;
    void record_cow(BlockId src_block_id, BlockId trg_block_id);
    std::vector<std::pair<BlockId, BlockId>> clear_cows();

private:
    std::vector<std::pair<BlockId, BlockId>> copy_on_writes_;
    const RefCounterProtocol& refcounter_;
};

// BlockPool: 블록 객체를 미리 할당하여 과도한 할당/해제를 방지
class BlockPool {
public:
    using BlockFactory = Block::Factory*;

    BlockPool(int block_size, BlockFactory create_block, BlockAllocator* allocator, int pool_size);
    
    std::shared_ptr<Block> init_block(
        std::shared_ptr<Block> prev_block,
        const std::vector<int>& token_ids,
        int block_size,
        int physical_block_id);

    void free_block(std::shared_ptr<Block> block);
    
    void increase_pool();

private:
    int block_size_;
    BlockFactory create_block_;
    BlockAllocator* allocator_;
    int pool_size_;
    std::deque<int> free_ids_;
    std::vector<std::shared_ptr<Block>> pool_;
};

// BlockList: 물리적 블록 ID에 대한 빠른 접근을 위한 최적화 클래스
class BlockList {
public:
    BlockList();
    
    void update(const std::vector<std::shared_ptr<Block>>& blocks);
    void append_token_ids(int block_index, const std::vector<int>& token_ids);
    void append(const std::shared_ptr<Block>& block);
    
    size_t size() const;
    std::shared_ptr<Block>& operator[](size_t index);
    const std::vector<std::shared_ptr<Block>>& list() const;
    const std::vector<int>& ids() const;
    void reset();

private:
    std::vector<std::shared_ptr<Block>> blocks_;
    std::vector<int> block_ids_;
};

// CacheMetricData: 블록의 캐시 메트릭을 추적
struct CacheMetricData {
    int num_completed_blocks = 0;
    float completed_block_cache_hit_rate = 0.0f;
    int num_incompleted_block_queries = 0;
    int num_incompleted_block_hit = 0;
    int block_size = 1000;

    void query(bool hit);
    float get_hit_rate() const;
};

// 마지막 블록부터 시작하여 모든 블록을 재귀적으로 가져옴
std::vector<std::shared_ptr<Block>> get_all_blocks_recursively(const std::shared_ptr<Block>& last_block);

#endif // PAGED_CORE_BLOCK_COMMON_H 