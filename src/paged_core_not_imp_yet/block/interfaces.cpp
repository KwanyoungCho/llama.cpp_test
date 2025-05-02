#include "interfaces.h"
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <optional>

// ---------- Default implementations for Block interface ----------

int Block::num_tokens_total() const {
    throw std::runtime_error("num_tokens_total not implemented");
}

bool Block::computed() const {
    throw std::runtime_error("computed not implemented");
}

void Block::set_computed(bool) {
    throw std::runtime_error("set_computed not implemented");
}

double Block::last_accessed() const {
    throw std::runtime_error("last_accessed not implemented");
}

void Block::set_last_accessed(double) {
    throw std::runtime_error("set_last_accessed not implemented");
}

int Block::content_hash() const {
    return 0;
}

// ---------- Default implementations for BlockAllocator interface ----------

std::shared_ptr<Block> BlockAllocator::allocate_mutable_block(
    std::shared_ptr<Block> /*prev_block*/) {
    throw std::runtime_error("allocate_mutable_block not implemented");
}

std::shared_ptr<Block> BlockAllocator::allocate_immutable_block(
    std::shared_ptr<Block> /*prev_block*/,
    const std::vector<int>& /*token_ids*/) {
    throw std::runtime_error("allocate_immutable_block not implemented");
}

std::vector<std::shared_ptr<Block>> BlockAllocator::allocate_immutable_blocks(
    std::shared_ptr<Block> /*prev_block*/,
    const std::vector<std::vector<int>>& /*block_token_ids*/) {
    throw std::runtime_error("allocate_immutable_blocks not implemented");
}

void BlockAllocator::free(std::shared_ptr<Block> block, bool keep_block_object) {
    throw std::runtime_error("free not implemented");
}

std::vector<std::shared_ptr<Block>> BlockAllocator::fork(std::shared_ptr<Block> /*last_block*/) {
    throw std::runtime_error("fork not implemented");
}

int BlockAllocator::get_num_total_blocks() const {
    throw std::runtime_error("get_num_total_blocks not implemented");
}

int BlockAllocator::get_num_free_blocks() const {
    throw std::runtime_error("get_num_free_blocks not implemented");
}

int BlockAllocator::get_physical_block_id(int /*absolute_id*/) {
    throw std::runtime_error("get_physical_block_id not implemented");
}

void BlockAllocator::swap_out(const std::vector<std::shared_ptr<Block>>& /*blocks*/) {
    throw std::runtime_error("swap_out not implemented");
}

void BlockAllocator::swap_in(const std::vector<std::shared_ptr<Block>>& /*blocks*/) {
    throw std::runtime_error("swap_in not implemented");
}

const std::unordered_set<int>& BlockAllocator::all_block_ids() const {
    throw std::runtime_error("all_block_ids not implemented");
}

std::vector<std::pair<int, int>> BlockAllocator::clear_copy_on_writes() {
    throw std::runtime_error("clear_copy_on_writes not implemented");
}

void BlockAllocator::mark_blocks_as_accessed(const std::vector<int>& /*block_ids*/, double /*now*/) {
    throw std::runtime_error("mark_blocks_as_accessed not implemented");
}

void BlockAllocator::mark_blocks_as_computed(const std::vector<int>& /*block_ids*/) {
    throw std::runtime_error("mark_blocks_as_computed not implemented");
}

std::vector<int> BlockAllocator::get_common_computed_block_ids(
    const std::vector<std::vector<int>>& /*computed_seq_block_ids*/) {
    throw std::runtime_error("get_common_computed_block_ids not implemented");
}

BlockId BlockAllocator::cow_block_if_not_appendable(const Block& /*block*/) {
    throw std::runtime_error("cow_block_if_not_appendable not implemented");
}

BlockId BlockAllocator::promote_to_immutable_block(const Block& /*block*/) {
    throw std::runtime_error("promote_to_immutable_block not implemented");
}

int BlockAllocator::get_num_full_blocks_touched(
    const std::vector<std::shared_ptr<Block>>& /*blocks*/) {
    throw std::runtime_error("get_num_full_blocks_touched not implemented");
}

float BlockAllocator::get_prefix_cache_hit_rate() {
    throw std::runtime_error("get_prefix_cache_hit_rate not implemented");
}

bool BlockAllocator::reset_prefix_cache() {
    throw std::runtime_error("reset_prefix_cache not implemented");
}

std::vector<int> BlockAllocator::find_cached_blocks_prefix(
    const std::vector<int>& /*block_hashes*/) {
    throw std::runtime_error("find_cached_blocks_prefix not implemented");
}

// ---------- Default implementations for DeviceAwareBlockAllocator interface ----------

std::shared_ptr<Block> DeviceAwareBlockAllocator::allocate_mutable_block(
    std::shared_ptr<Block> /*prev_block*/,
    Device /*device*/) {
    throw std::runtime_error("allocate_mutable_block not implemented");
}

std::shared_ptr<Block> DeviceAwareBlockAllocator::allocate_immutable_block(
    std::shared_ptr<Block> /*prev_block*/,
    const std::vector<int>& /*token_ids*/,
    Device /*device*/) {
    throw std::runtime_error("allocate_immutable_block not implemented");
}

std::vector<std::shared_ptr<Block>> DeviceAwareBlockAllocator::allocate_immutable_blocks(
    std::shared_ptr<Block> /*prev_block*/,
    const std::vector<std::vector<int>>& /*block_token_ids*/,
    Device /*device*/) {
    throw std::runtime_error("allocate_immutable_blocks not implemented");
}

int DeviceAwareBlockAllocator::get_num_free_blocks(Device /*device*/) {
    throw std::runtime_error("get_num_free_blocks not implemented");
}

int DeviceAwareBlockAllocator::get_num_total_blocks(Device /*device*/) {
    throw std::runtime_error("get_num_total_blocks not implemented");
}

void DeviceAwareBlockAllocator::free(Block* /*block*/) {
    throw std::runtime_error("free not implemented");
}

std::vector<std::shared_ptr<Block>> DeviceAwareBlockAllocator::fork(
    std::shared_ptr<Block> /*last_block*/) {
    throw std::runtime_error("fork not implemented");
}

const std::unordered_set<int>& DeviceAwareBlockAllocator::all_block_ids() const {
    throw std::runtime_error("all_block_ids not implemented");
}

std::vector<std::pair<int, int>> DeviceAwareBlockAllocator::clear_copy_on_writes() {
    throw std::runtime_error("clear_copy_on_writes not implemented");
}

void DeviceAwareBlockAllocator::mark_blocks_as_accessed(
    const std::vector<int>& /*block_ids*/,
    double /*now*/) {
    throw std::runtime_error("mark_blocks_as_accessed not implemented");
}

void DeviceAwareBlockAllocator::mark_blocks_as_computed(
    const std::vector<int>& /*block_ids*/) {
    throw std::runtime_error("mark_blocks_as_computed not implemented");
}

std::vector<int> DeviceAwareBlockAllocator::get_common_computed_block_ids(
    const std::vector<std::vector<int>>& /*computed_seq_block_ids*/) {
    throw std::runtime_error("get_common_computed_block_ids not implemented");
}

int DeviceAwareBlockAllocator::get_num_full_blocks_touched(
    const std::vector<std::shared_ptr<Block>>& /*blocks*/,
    Device /*device*/) {
    throw std::runtime_error("get_num_full_blocks_touched not implemented");
}

std::unordered_map<int, int> DeviceAwareBlockAllocator::swap(
    const std::vector<std::shared_ptr<Block>>& /*blocks*/,
    Device /*src_device*/,
    Device /*dst_device*/) {
    throw std::runtime_error("swap not implemented");
}

int DeviceAwareBlockAllocator::get_physical_block_id(Device /*device*/, int /*absolute_id*/) {
    throw std::runtime_error("get_physical_block_id not implemented");
}

std::shared_ptr<Block> DeviceAwareBlockAllocator::allocate_or_get_null_block() {
    throw std::runtime_error("allocate_or_get_null_block not implemented");
}

float DeviceAwareBlockAllocator::get_prefix_cache_hit_rate(Device /*device*/) {
    throw std::runtime_error("get_prefix_cache_hit_rate not implemented");
}

bool DeviceAwareBlockAllocator::reset_prefix_cache() {
    throw std::runtime_error("reset_prefix_cache not implemented");
}

std::vector<int> DeviceAwareBlockAllocator::find_cached_blocks_prefix(
    const std::vector<int>& /*block_hashes*/,
    Device /*device*/) {
    throw std::runtime_error("find_cached_blocks_prefix not implemented");
} 