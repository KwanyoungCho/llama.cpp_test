#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-memory.h"

#include "ggml-cpp.h"

#include <functional>
#include <set>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_ubatch;

/**
 * @brief KV 캐시의 기본 인터페이스 클래스
 * 
 * 모델이 이전에 처리한 토큰들의 Key와 Value 값을 저장하여
 * 추론 속도를 높이는 캐시 시스템의 기본 인터페이스.
 * llama_memory_i를 상속받아 메모리 관리 기능을 포함합니다.
 */
struct llama_kv_cache : public llama_memory_i {
    using llama_memory_i::llama_memory_i;

    // 현재 캐시에 저장된 총 토큰 수 반환
    virtual int32_t  get_n_tokens()   const = 0;
    
    // 현재 사용 중인 캐시 셀 수 반환 (통합 캐시 전용 기능)
    virtual uint32_t get_used_cells() const = 0; // TODO: remove, this is too-specific to the unified cache

    // 캐시가 시프트 연산을 지원하는지 여부 반환
    virtual bool get_can_shift() const = 0;

    // 캐시가 편집 가능한지 여부 반환 (시프트 지원 여부와 동일)
    bool get_can_edit() const override { return get_can_shift(); }
};

/**
 * @brief KV 캐시의 개별 셀 구조체
 * 
 * 캐시 내의 각 셀은 특정 위치의 토큰(들)에 대한 정보와
 * 해당 셀이 속한 시퀀스 ID 집합을 관리합니다.
 */
struct llama_kv_cell {
    llama_pos pos   = -1;        // 토큰 위치 (음수는 비어있음을 의미)
    llama_pos delta = 0;         // 위치 오프셋 조정값
    int32_t   src   = -1;        // 순환 상태 모델에서 상태 복사에 사용되는 소스 ID
    int32_t   tail  = -1;        // 순환 모델에서 최신 상태를 가리키는 테일 포인터

    std::set<llama_seq_id> seq_id;  // 이 셀에 연결된 시퀀스 ID 집합

    // 특정 시퀀스 ID가 이 셀에 있는지 확인
    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    // 셀이 비어있는지 확인 (시퀀스 없음)
    bool is_empty() const {
        return seq_id.empty();
    }

    // 두 셀이 동일한 시퀀스 집합을 가지는지 확인
    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

/**
 * @brief KV 캐시에서 찾은 슬롯 정보 구조체
 * 
 * find_slot 함수로 찾은 캐시 슬롯의 경계와 성공 여부를 저장합니다.
 */
struct llama_kv_cache_slot_info {
    std::pair<uint32_t, uint32_t> boundaries; // 슬롯 경계 [시작, 끝)
    bool found = false;                       // 슬롯 찾기 성공 여부

    explicit llama_kv_cache_slot_info(bool found_) : found{found_} {}
    llama_kv_cache_slot_info(uint32_t begin, uint32_t end) : boundaries{begin, end}, found{true} {}

    // bool 타입으로 변환 시 슬롯 찾기 성공 여부 반환
    operator bool() const { return found; }
};

/**
 * @brief 통합된 KV 캐시 구현 클래스 (링 버퍼 형태)
 * 
 * KV 데이터를 링 버퍼 형태로 관리하는 캐시 시스템.
 * 다양한 시퀀스의 캐시를 효율적으로 관리하고 재사용합니다.
 */
class llama_kv_cache_unified : public llama_kv_cache {
public:
    /**
     * @brief 모델 데이터 조회를 위한 콜백 구조체
     */
    struct callbacks {
        // RoPE(Rotary Position Embedding) 계수를 얻는 콜백 함수
        std::function<ggml_tensor * (uint32_t n_ctx_per_seq, int il)> get_rope_factors;
    };

    /**
     * @brief 생성자: 하이퍼파라미터와 콜백 저장
     */
    llama_kv_cache_unified(
            const llama_hparams & hparams,
            callbacks             cbs);

    virtual ~llama_kv_cache_unified() = default;

    /**
     * @brief 캐시 초기화 함수
     * 
     * @param model    모델 정보
     * @param cparams  컨텍스트 파라미터
     * @param type_k   K(key) 텐서의 데이터 타입
     * @param type_v   V(value) 텐서의 데이터 타입
     * @param kv_size  KV 캐시의 크기 (저장할 수 있는 토큰 수)
     * @param offload  GPU 오프로딩 사용 여부
     * @return 초기화 성공 여부
     */
    bool init(
            const llama_model & model,   // 모델 정보 (나중에 참조 제거 예정)
          const llama_cparams & cparams, // 컨텍스트 매개변수
                    ggml_type   type_k,  // K 텐서 데이터 타입
                    ggml_type   type_v,  // V 텐서 데이터 타입
                     uint32_t   kv_size, // 캐시 크기(셀 수)
                         bool   offload);// GPU 오프로딩 여부

    // 가상 함수 구현: 현재 캐시에 저장된 총 토큰 수 반환
    int32_t  get_n_tokens()   const override;
    // 가상 함수 구현: 현재 사용 중인 캐시 셀 수 반환
    uint32_t get_used_cells() const override;

    // 캐시가 사용하는 총 메모리 크기 반환
    size_t total_size() const;

    // 캐시에 저장된 최대 위치(pos) 값 반환
    llama_pos pos_max() const;

    // 캐시 전체 초기화 (모든 셀 비움)
    void clear() override;
    // 캐시 조각 모음 수행 (연속적인 메모리 사용을 위해)
    void defrag() override;

    // 특정 시퀀스 ID의 특정 범위 위치 토큰 제거
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    // 한 시퀀스의 상태를 다른 시퀀스로 복사
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    // 특정 시퀀스만 유지하고 나머지 제거
    void seq_keep(llama_seq_id seq_id) override;
    // 특정 시퀀스의 위치 값에 델타 추가
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) override;
    // 특정 시퀀스의 위치 값을 d로 나눔
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    // 특정 시퀀스의 최대 위치 값 반환
    llama_pos seq_pos_max(llama_seq_id seq_id) override;

    // 캐시가 시프트 연산을 지원하는지 여부 반환
    bool get_can_shift() const override;

    /**
     * @brief 캐시에서 n_tokens 크기의 빈 슬롯 찾기
     * 
     * 캐시 헤드를 업데이트하고 찾은 슬롯 정보 반환
     * 성공 시 캐시 헤드(head)가 슬롯의 첫 셀을 가리키는 것이 중요
     * 
     * @param batch 처리할 마이크로 배치
     * @return 찾은 슬롯 정보
     */
    llama_kv_cache_slot_info find_slot(const llama_ubatch & batch);

    // 컨텍스트 파라미터에 따른 패딩 값 계산
    uint32_t get_padding(const llama_cparams & cparams) const;

    // 현재 사용 중인 최대 셀 인덱스 찾기
    uint32_t cell_max() const;

    // K 텐서가 사용하는 메모리 크기 계산
    size_t size_k_bytes() const;
    // V 텐서가 사용하는 메모리 크기 계산
    size_t size_v_bytes() const;

    /**
     * @brief 조각 모음 관련 정보 구조체
     */
    struct {
        std::vector<uint32_t> ids;  // 조각 모음 시 사용할 ID 목록
    } defrag_info;

    /**
     * @brief 조각 모음 준비 함수
     * 
     * @param n_max_nodes 최대 노드 수
     * @return 셀이 이동되었는지 여부
     */
    bool defrag_prepare(int32_t n_max_nodes);

    /**
     * @brief 캐시 상태 저장/로드 함수
     */
    // 캐시 상태 저장 (특정 시퀀스 또는 전체)
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const;
    // 캐시 상태 로드 (특정 시퀀스 또는 전체)
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1);

    // 멤버 변수들

    const llama_hparams & hparams;  // 모델 하이퍼파라미터 참조

    callbacks cbs;  // 콜백 함수 모음

    bool has_shift = false;  // 시프트 연산이 수행되었는지 여부
    bool do_defrag = false;  // 조각 모음이 필요한지 여부

    // 순환 모델 여부 (Mamba, RWKV 등)
    // 순환 모델에서는 한 셀이 여러 과거 토큰의 상태를 보유할 수 있음
    bool recurrent = false;  

    bool v_trans   = true;   // V 텐서가 전치(transpose)되었는지 여부
    bool can_shift = false;  // 시프트 연산 지원 여부

    // 캐시 링 버퍼의 현재 헤드 위치
    // 주의: 헤드 값은 빈 슬롯 검색 최적화뿐 아니라 
    // llama_decode_impl에서도 사용되므로 슬롯 할당 후 함부로 변경 불가
    uint32_t head = 0;
    uint32_t size = 0;  // 캐시 크기 (총 셀 수)
    uint32_t used = 0;  // 사용 중인 셀 수 (적어도 하나의 seq_id가 있는 셀)

    // 각 그래프 빌드 전에 계산되는 유효 셀 수
    uint32_t n = 0;

    // 캐시 셀 배열
    std::vector<llama_kv_cell> cells;

    // 레이어별 K, V 텐서 배열
    std::vector<ggml_tensor *> k_l;  // 각 레이어의 K(key) 텐서
    std::vector<ggml_tensor *> v_l;  // 각 레이어의 V(value) 텐서

private:
    // K, V 텐서의 데이터 타입 (기본값: F16)
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    // GGML 컨텍스트와 백엔드 버퍼
    std::vector<ggml_context_ptr>        ctxs;  // 각 버퍼 타입별 GGML 컨텍스트
    std::vector<ggml_backend_buffer_ptr> bufs;  // 각 컨텍스트의 백엔드 버퍼

    // 메타데이터 저장 함수
    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    // 데이터 저장 함수
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    // 메타데이터 로드 함수
    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    // 데이터 로드 함수
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

// TODO: 임시로 llama_kv_cache_unified 재사용 -- 순환 캐시 구현 및 통합 캐시 단순화 필요
//class llama_kv_cache_recurrent : public llama_kv_cache_unified {
//public:
//    using llama_kv_cache_unified::llama_kv_cache_unified;
//};

/**
 * @brief KV 캐시 상태 복원 관리 구조체
 * 
 * 미래 복구를 위해 KV 캐시 상태를 저장합니다.
 * llama_kv_cache_find_slot 변경사항을 롤백하는 데 사용됩니다.
 */
struct llama_kv_slot_restorer {
    // 캐시 상태 저장 구조체
    struct llama_kv_cache_state {
        uint32_t head = 0;  // 캐시 헤드 위치
        uint32_t n    = 0;  // 유효 셀 수
    } old_state;  // 이전 상태 저장

    // 비순환 모델 전용: 복원할 슬롯 목록
    std::vector<std::pair<uint32_t, uint32_t>> slot_boundaries;

    bool do_restore = false;  // 복원 필요 여부

    llama_kv_cache_unified & cache;  // 참조하는 캐시 객체

    // 생성자: 현재 캐시 상태 저장
    explicit llama_kv_slot_restorer(llama_kv_cache_unified & cache) : cache(cache) {
        old_state.head = cache.head;
        old_state.n    = cache.n;
    }

    /**
     * @brief 슬롯 정보 저장 (나중에 복원하기 위해)
     * 
     * @param slot 저장할 슬롯 정보
     */
    void save(const llama_kv_cache_slot_info & slot) {
        if (slot) {
            do_restore = true;
            if (slot.boundaries.first != slot.boundaries.second) {
                slot_boundaries.push_back(slot.boundaries);
            }
        }
    }

    /**
     * @brief KV 캐시 상태 복원 및 모든 find_slot 호출 변경사항 롤백
     * 
     * 명시적으로 호출해야 합니다.
     */
    void restore() {
        if (do_restore) {
            cache.head = old_state.head;
            cache.n    = old_state.n;

            if (cache.recurrent) { // Mamba나 RWKV 같은 순환 모델은 상태를 부분적으로 지울 수 없음
                cache.seq_rm(-1, -1, -1);
            } else {
                for (auto & slot : slot_boundaries) {
                    cache.seq_rm(-1, slot.first, slot.second);
                }
            }
        }
    }
};

/**
 * @brief 유틸리티 함수들 (나중에 llama_kv_cache의 공개 API가 될 수 있음)
 */

// 캐시에 저장된 총 토큰 수 반환
int32_t llama_kv_cache_n_tokens(const llama_kv_cache * kv);

// 현재 사용 중인 캐시 셀 수 반환
int32_t llama_kv_cache_used_cells(const llama_kv_cache * kv);

// 캐시 전체 초기화
void llama_kv_cache_clear(llama_kv_cache * kv);

// 특정 시퀀스 ID의 특정 범위 위치 토큰 제거
bool llama_kv_cache_seq_rm(
        llama_kv_cache * kv,     // 대상 KV 캐시
          llama_seq_id   seq_id, // 제거할 시퀀스 ID (-1: 모든 시퀀스)
             llama_pos   p0,     // 시작 위치 (포함)
             llama_pos   p1);    // 끝 위치 (제외)

// 한 시퀀스의 상태를 다른 시퀀스로 복사
void llama_kv_cache_seq_cp(
        llama_kv_cache * kv,        // 대상 KV 캐시
          llama_seq_id   seq_id_src,// 소스 시퀀스 ID
          llama_seq_id   seq_id_dst,// 대상 시퀀스 ID
             llama_pos   p0,        // 시작 위치 (포함)
             llama_pos   p1);       // 끝 위치 (제외)

// 특정 시퀀스만 유지하고 나머지 제거
void llama_kv_cache_seq_keep(llama_kv_cache * kv, llama_seq_id seq_id);

// 특정 시퀀스의 위치 값에 델타 추가
void llama_kv_cache_seq_add(
        llama_kv_cache * kv,     // 대상 KV 캐시
          llama_seq_id   seq_id, // 대상 시퀀스 ID
             llama_pos   p0,     // 시작 위치 (포함) 
             llama_pos   p1,     // 끝 위치 (제외)
             llama_pos   delta); // 추가할 오프셋 값

// 특정 시퀀스의 위치 값을 d로 나눔
void llama_kv_cache_seq_div(
        llama_kv_cache * kv,     // 대상 KV 캐시
          llama_seq_id   seq_id, // 대상 시퀀스 ID
             llama_pos   p0,     // 시작 위치 (포함)
             llama_pos   p1,     // 끝 위치 (제외)
                   int   d);     // 나눌 값

// 특정 시퀀스의 최대 위치 값 반환
llama_pos llama_kv_cache_seq_pos_max(llama_kv_cache * kv, llama_seq_id seq_id);

// 캐시 조각 모음 수행
void llama_kv_cache_defrag(llama_kv_cache * kv);

// 캐시가 시프트 연산을 지원하는지 여부 반환
bool llama_kv_cache_can_shift(const llama_kv_cache * kv);

/**
 * @brief KV 캐시 뷰 관련 함수
 */

// KV 캐시 뷰 초기화 (지정된 최대 시퀀스 수로)
llama_kv_cache_view llama_kv_cache_view_init(const llama_kv_cache & kv, int32_t n_seq_max);

// KV 캐시 뷰 업데이트 (캐시 변경 사항 반영)
void llama_kv_cache_view_update(llama_kv_cache_view * view, const llama_kv_cache * kv);
