#pragma once

#include <vector>
#include <iterator>

namespace paged_core {

template<typename T>
class chunk_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::vector<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    chunk_iterator(const std::vector<T>& vec, size_t chunk_size, size_t pos = 0)
        : _vec(vec), _chunk_size(chunk_size), _pos(pos) {
        if (_chunk_size == 0) _chunk_size = 1;
    }

    value_type operator*() const {
        if (_pos >= _vec.size()) {
            return value_type();
        }
        size_t end = std::min(_pos + _chunk_size, _vec.size());
        return std::vector<T>(_vec.begin() + _pos, _vec.begin() + end);
    }

    chunk_iterator& operator++() {
        if (_pos < _vec.size()) {
            _pos += _chunk_size;
        }
        return *this;
    }

    chunk_iterator operator++(int) {
        chunk_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const chunk_iterator& other) const {
        return _pos >= _vec.size() && other._pos >= other._vec.size();
    }

    bool operator!=(const chunk_iterator& other) const {
        return !(*this == other);
    }

private:
    const std::vector<T>& _vec;
    size_t _chunk_size;
    size_t _pos;
};

template<typename T>
class chunk_list {
public:
    chunk_list(const std::vector<T>& vec, size_t chunk_size)
        : _vec(vec), _chunk_size(chunk_size) {}

    chunk_iterator<T> begin() const {
        return chunk_iterator<T>(_vec, _chunk_size);
    }

    chunk_iterator<T> end() const {
        return chunk_iterator<T>(_vec, _chunk_size, _vec.size());
    }

private:
    const std::vector<T>& _vec;
    size_t _chunk_size;
};

} // namespace paged_core 