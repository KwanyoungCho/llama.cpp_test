/* NEW FILE */
#ifndef BLOCK_MANAGER_H
#define BLOCK_MANAGER_H

#include <vector>
#include <cstddef>
#include <memory>

// 이 클래스는 vllm/vllm/core/block_manager.py로부터 포팅된 paged attention을 위한 블록 관리 클래스입니다.
// 기존 구현에서 필요없는 파이썬 종속성은 제거하고, 기본적인 블록 할당 및 캐시 관리 기능만 구현합니다.
class BlockManager {
public:
    explicit BlockManager(size_t block_size);
    ~BlockManager();

    // 새로운 블록을 할당합니다.
    void allocate_block();

    // 블록 캐시를 관리합니다.
    void manage_cache();

private:
    size_t block_size_;
    std::vector<void*> blocks_;
};

#endif // BLOCK_MANAGER_H
