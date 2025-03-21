#include "utils.h"
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace block_utils {

std::string tokenIdsToString(const std::vector<int>& tokenIds) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        oss << tokenIds[i];
        if (i != tokenIds.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

void printBlockInfo(const std::shared_ptr<Block>& block) {
    if (!block) {
        std::cout << "Block is null" << std::endl;
        return;
    }
    std::cout << "Block ID: " << block->getBlockId() << std::endl;
    std::cout << "Token IDs: " << tokenIdsToString(block->getTokenIds()) << std::endl;
    std::cout << "Empty slots: " << block->numEmptySlots() << std::endl;
    std::cout << "Is full: " << (block->isFull() ? "Yes" : "No") << std::endl;
    if (block->getPrevBlock()) {
        std::cout << "Has previous block" << std::endl;
    } else {
        std::cout << "No previous block" << std::endl;
    }
}

static const std::string STR_NOT_IMPL_ENC_DEC_SWA = "Sliding-window attention (SWA) is not implemented for encoder/decoder models.";
static const std::string STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE = "Prefix caching is not implemented for encoder/decoder models.";

void checkNoCachingOrSwaForBlockMgrEncdec(const BlockManager& blockMgr, const SequenceGroup& seqGroup) {
    if (seqGroup.is_encoder_decoder()) {
        if (blockMgr.max_block_sliding_window.has_value()) {
            throw std::runtime_error(STR_NOT_IMPL_ENC_DEC_SWA);
        }
        if (blockMgr.enable_caching) {
            throw std::runtime_error(STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE);
        }
    }
}

} // namespace block_utils 