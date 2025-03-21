#ifndef PAGED_CORE_BLOCK_UTILS_H
#define PAGED_CORE_BLOCK_UTILS_H

#include "interfaces.h"
#include <vector>
#include <string>
#include <memory>
#include <optional>

namespace block_utils {

// 함수 선언

// tokenIdsToString: 토큰 ID들(vector<int>)을 문자열 형식 "[1, 2, 3]"으로 변환하는 함수.
std::string tokenIdsToString(const std::vector<int>& tokenIds);

// printBlockInfo: Block의 ID, token ID들, 빈 슬롯 수, full 여부, 이전 블록 존재여부를 출력하는 함수.
void printBlockInfo(const std::shared_ptr<Block>& block);

// 추가적인 기능을 위한 타입 선언 (Python utils.py와 유사한 기능 구현을 위해)
// BlockManager: max_block_sliding_window와 enable_caching 멤버를 가짐.
struct BlockManager {
    std::optional<int> max_block_sliding_window;
    bool enable_caching;
};

// SequenceGroup: is_encoder_decoder() 메서드를 제공하는 추상 클래스.
class SequenceGroup {
public:
    virtual bool is_encoder_decoder() const = 0;
    virtual ~SequenceGroup() = default;
};

// checkNoCachingOrSwaForBlockMgrEncdec: 인코더/디코더 모델에 대해 prefix caching과 sliding-window attention (SWA)가 지원되지 않음을 검사.
// 만약 seqGroup.is_encoder_decoder()가 true인 경우, blockMgr의 max_block_sliding_window가 설정되어 있거나 enable_caching이 true이면
// 각각 NotImplementedError에 해당하는 메시지를 담은 std::runtime_error 예외를 발생시킵니다.
void checkNoCachingOrSwaForBlockMgrEncdec(const BlockManager& blockMgr, const SequenceGroup& seqGroup);

} // namespace block_utils

#endif // PAGED_CORE_BLOCK_UTILS_H 