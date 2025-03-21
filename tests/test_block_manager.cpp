/* File: llama.cpp/tests/test_block_manager.cpp */
#include <iostream>
#include "block_manager.h"
#include <vector>

int main() {
    std::cout << "Creating BlockManager with 3 blocks" << std::endl;
    BlockManager bm(3);
    
    // Allocate three blocks
    std::cout << "Allocating three blocks:" << std::endl;
    int block0 = bm.allocateBlock();
    int block1 = bm.allocateBlock();
    int block2 = bm.allocateBlock();
    std::cout << "Allocated blocks: " << block0 << ", " << block1 << ", " << block2 << std::endl;
    
    // Allocate a fourth block to trigger eviction
    std::cout << "Allocating a fourth block to trigger eviction:" << std::endl;
    int block3 = bm.allocateBlock();
    std::cout << "Fourth block allocated (evicted block): " << block3 << std::endl;
    
    // Free block1 and allocate again
    std::cout << "Freeing block " << block1 << std::endl;
    bm.freeBlock(block1);
    int block4 = bm.allocateBlock();
    std::cout << "Allocated block after freeing: " << block4 << std::endl;
    
    // Print final block allocation statuses
    std::vector<bool> status = bm.getBlockStatus();
    std::cout << "Final block status:" << std::endl;
    for (size_t i = 0; i < status.size(); ++i) {
        std::cout << "Block " << i << ": " << (status[i] ? "Allocated" : "Free") << std::endl;
    }
    
    // 추가 엣지 테스트
    std::cout << "\n========== 추가 엣지 테스트 ==========" << std::endl;
    // 새로운 BlockManager 인스턴스를 생성 (2개의 블록)
    BlockManager bm2(2);
    std::cout << "새로운 BlockManager (2블록) 생성." << std::endl;
    
    // 아직 할당되지 않은 블록 해제 시도 (에러 없이 처리되어야 함)
    std::cout << "할당되지 않은 블록 (인덱스 1) 해제 시도:" << std::endl;
    bm2.freeBlock(1);
    
    // 연속할당 테스트: 3번 할당하여 eviction 확인
    std::cout << "연속적으로 3번 할당하여 eviction 테스트:" << std::endl;
    int bA = bm2.allocateBlock();
    int bB = bm2.allocateBlock();
    int bC = bm2.allocateBlock();
    std::cout << "할당된 블록들: " << bA << ", " << bB << ", " << bC << std::endl;
    
    // double free 테스트
    std::cout << "double free 테스트:" << std::endl;
    bm2.freeBlock(bB);
    bm2.freeBlock(bB);
    
    // 새로운 BlockManager의 최종 상태 출력
    std::vector<bool> status2 = bm2.getBlockStatus();
    std::cout << "새 BlockManager의 최종 상태:" << std::endl;
    for (size_t i = 0; i < status2.size(); ++i) {
        std::cout << "Block " << i << ": " << (status2[i] ? "Allocated" : "Free") << std::endl;
    }
    
    return 0;
} 