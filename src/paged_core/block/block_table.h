#ifndef BLOCK_TABLE_H
#define BLOCK_TABLE_H

#include <unordered_map>
#include <memory>
#include <vector>
#include "interfaces.h"
#include "common.h"           // for BlockList
#include "cpu_gpu_block_allocator.h"

// BlockTable manages a mapping from block IDs to Block pointers.
class BlockTable {
public:
    // 기본 매핑 기능 생성자 (기존 기능)
    BlockTable();
    
    // 확장된 기능을 위한 생성자: block_size, allocator, 초기 블록 리스트, max_block_sliding_window (-1이면 사용 안함)
    BlockTable(int block_size, CPUGPUBlockAllocator* allocator,
               const std::vector<std::shared_ptr<Block>>& init_blocks = {},
               int max_block_sliding_window = -1);
               
    ~BlockTable();

    // [기본 매핑 기능]
    void addBlock(int blockId, std::shared_ptr<Block> block);
    std::shared_ptr<Block> getBlock(int blockId) const;
    void removeBlock(int blockId);
    void clear();
    void updateBlock(int oldBlockId, int newBlockId, std::shared_ptr<Block> block);

    // [확장 기능: BlockTableFull과 유사한 기능]
    // Allocates blocks for the given token sequence.
    void allocate(const std::vector<int>& token_ids);
    
    // Updates the table with newly allocated blocks.
    void update(const std::vector<std::shared_ptr<Block>>& blocks);
    
    // Appends token IDs to existing blocks.
    // num_computed_slots: computed token count (use -1 if not applicable)
    void append_token_ids(const std::vector<int>& token_ids, int num_lookahead_slots = 0, int num_computed_slots = -1);
                          
    // Ensures that there are at least 'num_empty_slots' free slots in the block table.
    void ensure_num_empty_slots(int num_empty_slots);
    
    // Frees all blocks managed by this table.
    void free_all();
    
    // Returns a list of physical block IDs.
    std::vector<int> physical_block_ids() const;
    
    // Given a full sequence of token IDs, returns the postfix that hasn't been appended.
    std::vector<int> get_unseen_token_ids(const std::vector<int>& sequence_token_ids) const;
    
    // Total number of tokens stored in the BlockTable.
    int num_full_slots() const;
    
    // Determines how many blocks will be affected by appending token_ids (plus optional lookahead slots).
    int get_num_blocks_touched_by_append_slots(const std::vector<int>& token_ids,
                                                 int num_lookahead_slots) const;
    
    // Returns the underlying list of blocks.
    std::vector<std::shared_ptr<Block>> blocks() const;

private:
    // --- 기존 기본 매핑 멤버 --- 
    std::unordered_map<int, std::shared_ptr<Block>> table_;

    // --- 확장 기능 위한 멤버 ---
    int _block_size;                            // 한 블록당 최대 토큰 수
    CPUGPUBlockAllocator* _allocator;           // 블록 할당자
    BlockList _block_list;                      // BlockList: 블록들을 관리하는 컨테이너
    int _num_full_slots;                        // 현재 저장된 토큰 수
    int _max_block_sliding_window;              // sliding window를 사용할 경우 유지할 블록 수 (-1이면 사용 안함)

    // Helper: 정수 a를 b로 나누고 올림 처리
    int cdiv(int a, int b) const { return (a + b - 1) / b; }
    
    // Helper: 토큰 ID들을 블록 크기 단위로 분할하여 벡터로 반환
    std::vector<std::vector<int>> chunk_token_blocks_for_append(const std::vector<int>& token_ids) const;
    
    // Helper: 블록에 저장된 모든 토큰 ID의 총합 계산
    int get_num_token_ids() const;
};

#endif // BLOCK_TABLE_H 