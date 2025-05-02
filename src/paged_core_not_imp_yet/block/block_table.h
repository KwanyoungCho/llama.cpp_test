#ifndef BLOCK_TABLE_H
#define BLOCK_TABLE_H

#include <memory>
#include <vector>
#include "interfaces.h"
#include "common.h"           // for BlockList


/**
 * A class to manage blocks for a specific sequence.
 *
 * The BlockTable maps a sequence of tokens to a list of blocks, where each
 * block represents a contiguous memory allocation for a portion of the 
 * sequence. The blocks are managed by a DeviceAwareBlockAllocator, which is
 * responsible for allocating and freeing memory for the blocks.
 */
class BlockTable {
public:
    // 기본 생성자: 기본 매핑 기능만 사용할 경우
    BlockTable();
    
    // 확장 기능 생성자
    BlockTable(int block_size, DeviceAwareBlockAllocator* block_allocator,
               const std::vector<std::shared_ptr<Block>>& _blocks = std::vector<std::shared_ptr<Block>>(),
               int max_block_sliding_window = -1);
               
    ~BlockTable();

    // 정적 메서드: 주어진 토큰 ID와 블록 크기를 고려하여 필요한 블록 수 계산
    static int get_num_required_blocks(const std::vector<int>& token_ids,
                                       int block_size,
                                       int num_lookahead_slots = 0);

    // [확장 기능]
    void allocate(const std::vector<int>& token_ids, Device device);
    void update(const std::vector<std::shared_ptr<Block>>& blocks);
    void append_token_ids(const std::vector<int>& token_ids,
                         int num_lookahead_slots,
                         int num_computed_slots);
    void ensure_num_empty_slots(int num_empty_slots);
    void free();
    std::vector<int> physical_block_ids() const;
    std::vector<int> get_unseen_token_ids(const std::vector<int>& sequence_token_ids) const;
    int num_full_slots() const;
    int get_num_blocks_touched_by_append_slots(const std::vector<int>& token_ids,
                                              int num_lookahead_slots) const;
    
    // Properties (getter 메서드)
    std::vector<std::shared_ptr<Block>> blocks() const;
    bool _is_allocated() const;
    int _num_empty_slots() const;

    std::shared_ptr<BlockTable> fork();

private:
    // --- Private Helper Methods ---
    std::vector<std::vector<int>> _chunk_token_blocks_for_append(const std::vector<int>& token_ids) const;
    int _get_num_token_ids() const;
    std::vector<int> _get_all_token_ids() const;
    std::vector<std::shared_ptr<Block>> _allocate_blocks_for_token_ids(
        std::shared_ptr<Block> prev_block,
        const std::vector<int>& token_ids,
        Device device);

    // --- Member Variables ---
    int _block_size;
    DeviceAwareBlockAllocator* _allocator;
    BlockList _blocks;
    int _num_full_slots;
    int _max_block_sliding_window;
};

#endif // BLOCK_TABLE_H 