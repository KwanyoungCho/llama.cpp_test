#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <stdexcept>

static const llama_kv_cache_slot_info llama_kv_cache_slot_info_failed{false};

// llama_kv_cache_unified 클래스 생성자: hyperparameters와 콜백 함수 저장
llama_kv_cache_unified::llama_kv_cache_unified(const llama_hparams & hparams, callbacks cbs) : hparams(hparams), cbs(std::move(cbs)) {
}

/**
 * LLaMA 모델의 통합 KV(Key-Value) 캐시를 초기화하는 함수
 * 
 * @param model   초기화할 LLaMA 모델
 * @param cparams 모델 컨텍스트 매개변수
 * @param type_k  Key 텐서의 데이터 타입 (ggml_type)
 * @param type_v  Value 텐서의 데이터 타입 (ggml_type)
 * @param kv_size KV 캐시의 크기 (토큰 수)
 * @param offload 계산을 GPU로 오프로드할지 여부
 * 
 * @return 초기화 성공 여부
 */
bool llama_kv_cache_unified::init(
        const llama_model & model,
      const llama_cparams & cparams,
                ggml_type   type_k,
                ggml_type   type_v,
                 uint32_t   kv_size,
                     bool   offload) {
    const int32_t n_layer = hparams.n_layer;

    // 캐시 시프트 상태 초기화
    has_shift = false;

    // 모델 특성 설정
    // 순환 신경망(RNN) 모델인지 확인 - RNN은 다른 캐싱 메커니즘 필요
    recurrent = llama_model_is_recurrent(&model);
    // v_trans: value 텐서 전치 여부 (순환 모델이 아니고 flash attention을 사용하지 않을 때 활성화)
    // 전치는 메모리 접근 패턴 최적화에 사용됨
    v_trans   = !recurrent && !cparams.flash_attn;
    // 시프트 작업 지원 여부 (순환 모델이 아니고 DeepSeek2 아키텍처가 아닐 때 가능)
    // 시프트 작업은 캐시의 내용을 이동시켜 새 토큰을 위한 공간 확보
    can_shift = !recurrent && model.arch != LLM_ARCH_DEEPSEEK2; // MLA(Multi-Layer Attention) 때문에 DeepSeek2에서는 지원되지 않음

    LLAMA_LOG_INFO("%s: kv_size = %d, offload = %d, type_k = '%s', type_v = '%s', n_layer = %d, can_shift = %d\n",
            __func__, kv_size, offload, ggml_type_name(type_k), ggml_type_name(type_v), n_layer, can_shift);

    // 캐시 상태 초기화
    head = 0;  // 캐시 검색 시작 위치 (링 버퍼의 현재 시작점)
    size = kv_size;  // 캐시의 총 크기 (최대 저장 가능한 토큰 수)
    used = 0;  // 현재 사용 중인 셀 수 (저장된 토큰 수)

    // 텐서 데이터 타입 저장
    this->type_k = type_k;
    this->type_v = type_v;

    // 캐시 셀 배열 초기화 - 각 셀은 하나의 토큰에 대한 정보를 저장
    cells.clear();
    cells.resize(kv_size);

    // 각 버퍼 타입별로 ggml 컨텍스트 생성 (CPU/GPU별 별도 컨텍스트 관리)
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // 새로운 버퍼 타입에 대한 ggml 컨텍스트 초기화
            // 이 컨텍스트는 해당 backend(CPU/GPU)에서 텐서를 생성하는 데 사용됨
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer*ggml_tensor_overhead()),  // 레이어당 K,V 텐서 2개의 오버헤드 할당
                /*.mem_buffer =*/ NULL,  // 메모리는 나중에 할당
                /*.no_alloc   =*/ true,  // 즉시 메모리 할당하지 않음
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);  // 컨텍스트 목록에 저장 (나중에 정리하기 위함)

            return ctx;
        }

        return it->second;
    };

    // 각 레이어별 K, V 텐서 벡터 준비
    k_l.reserve(n_layer);
    v_l.reserve(n_layer);

    // 각 레이어에 대해 K, V 텐서 생성
    for (int i = 0; i < n_layer; i++) {
        // GQA(Grouped-Query Attention)를 위한 임베딩 차원 계산
        // n_embd_k_s와 n_embd_v_s는 추가 상태 정보를 위한 차원
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        const char * dev_name = "CPU";

        // 버퍼 타입 결정 (오프로딩 사용 시 GPU, 아니면 CPU)
        ggml_backend_buffer_type_t buft;
        if (offload) {
            // 오프로드 설정 시 해당 레이어에 지정된 디바이스 사용 (보통 GPU)
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        } else {
            // 오프로드가 아닐 경우 CPU 사용
            buft = ggml_backend_cpu_buffer_type();
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: n_embd_k_gqa = %d, n_embd_v_gqa = %d, dev = %s\n", __func__,
                i, n_embd_k_gqa, n_embd_v_gqa, dev_name);

        // 해당 버퍼 타입에 맞는 컨텍스트 가져오기
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
            return false;
        }

        // K, V 텐서 생성 및 이름 지정
        // 1D 텐서로 생성하여 연속된 메모리 공간에 저장
        // 크기는 [임베딩 차원 × 캐시 크기]로, 모든 토큰에 대한 키/값을 저장
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);  // 디버깅을 위한 텐서 이름 지정
        ggml_format_name(v, "cache_v_l%d", i);
        k_l.push_back(k);  // 레이어별 K 텐서 저장
        v_l.push_back(v);  // 레이어별 V 텐서 저장
    }

    // 텐서 할당 및 버퍼 초기화 (NaN 방지)
    for (auto it : ctx_map) {
        auto * buft = it.first;  // 버퍼 타입 (CPU 또는 GPU)
        auto * ctx  = it.second; // 해당 버퍼 타입의 ggml 컨텍스트

        // 컨텍스트의 모든 텐서를 위한 백엔드 버퍼 할당
        // 실제 메모리 할당이 이루어지는 부분
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
            return false;
        }
        // 버퍼를 0으로 초기화 - 미초기화 메모리로 인한 NaN 문제 방지
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        bufs.emplace_back(buf);  // 할당된 버퍼 저장 (나중에 정리하기 위함)
    }

    return true;  // 초기화 성공
}

/**
 * 캐시에 저장된 총 토큰 수 반환
 * @return 저장된 토큰 수
 */
int32_t llama_kv_cache_unified::get_n_tokens() const {
    int32_t result = 0;

    // 모든 셀에 저장된 시퀀스 ID 수 합산
    for (uint32_t i = 0; i < size; i++) {
        result += cells[i].seq_id.size();
    }

    return result;
}

/**
 * 현재 사용 중인 캐시 셀 수 반환
 * @return 사용 중인 셀 수
 */
uint32_t llama_kv_cache_unified::get_used_cells() const {
    return used;
}

/**
 * KV 캐시가 사용하는 총 메모리 크기 계산
 * @return 총 메모리 크기(바이트)
 */
size_t llama_kv_cache_unified::total_size() const {
    size_t size = 0;
    // 모든 버퍼 크기 합산
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

/**
 * 캐시에 저장된 최대 위치(pos) 값 반환
 * @return 최대 위치 값 (없으면 -1)
 */
llama_pos llama_kv_cache_unified::pos_max() const {
    llama_pos pos_max = -1;
    for (const auto & cell : cells) {
        pos_max = std::max(pos_max, cell.pos);
    }

    return pos_max;
}

/**
 * KV 캐시 전체 초기화
 * 모든 셀의 상태를 리셋하고 버퍼를 0으로 채움
 */
void llama_kv_cache_unified::clear() {
    // 모든 셀 초기화
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;  // 위치 무효화
        cells[i].seq_id.clear();  // 시퀀스 ID 제거
        cells[i].src = -1;  // 소스 정보 제거
        cells[i].tail = -1;  // 테일 정보 제거
    }
    head = 0;  // 헤드 위치 초기화
    used = 0;  // 사용 중인 셀 수 초기화

    // 모든 버퍼 내용 0으로 초기화
    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

/**
 * 특정 시퀀스 ID의 특정 범위 위치 토큰 제거
 * 
 * @param seq_id 제거할 시퀀스 ID
 * @param p0     제거 시작 위치 (포함, 기본값 0)
 * @param p1     제거 끝 위치 (제외, 기본값 최대값)
 * @return       제거 성공 여부
 */
bool llama_kv_cache_unified::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    // 기본값 설정
    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // Mamba나 RWKV 같은 순환 모델은 상태를 부분적으로 지울 수 없음
    if (recurrent) {
        if (seq_id >= (int64_t) size) {
            // 치명적일 수 있는 오류
            return false;
        }
        if (0 <= seq_id) {
            int32_t & tail_id = cells[seq_id].tail;
            if (tail_id >= 0) {
                const llama_kv_cell & cell = cells[tail_id];
                // 부분 교차는 유효하지 않음
                if ((0 < p0 && p0 <= cell.pos) || (0 < p1 && p1 <= cell.pos)) {
                    return false;
                }
                // 삭제될 tail들 무효화
                if (p0 <= cell.pos && cell.pos < p1) {
                    tail_id = -1;
                }
            }
        } else {
            // seq_id가 음수이면 범위는 전체를 포함하거나 아무것도 포함하지 않아야 함
            if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
                return false;
            }
        }
    }

    // 모든 셀을 순회하며 조건에 맞는 시퀀스 ID 제거
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                // 음수 seq_id는 모든 시퀀스 ID 제거
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                // 특정 시퀀스 ID만 제거
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            // 셀이 비어있으면 완전히 초기화
            if (cells[i].is_empty()) {
                // 사용 중인 셀 수 감소
                if (cells[i].pos >= 0) {
                    used--;
                }

                cells[i].pos = -1;
                cells[i].src = -1;

                // 새 헤드 위치 갱신
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // 빈 슬롯이 생겼고 현재 헤드보다 앞에 있으면 헤드 위치 조정
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

/**
 * 한 시퀀스의 상태를 다른 시퀀스로 복사
 * 
 * @param seq_id_src 소스 시퀀스 ID
 * @param seq_id_dst 대상 시퀀스 ID
 * @param p0         복사 시작 위치 (포함, 기본값 0)
 * @param p1         복사 끝 위치 (제외, 기본값 최대값)
 */
void llama_kv_cache_unified::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // 소스와 대상이 같으면 작업 불필요
    if (seq_id_src == seq_id_dst) {
        return;
    }

    // 기본값 설정
    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // 순환 모델(RNN)의 경우 특별 처리
    if (recurrent) {
        if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
            llama_kv_cell & tail_src = cells[seq_id_src];
            llama_kv_cell & tail_dst = cells[seq_id_dst];
            
            // 대상 시퀀스에 이미 상태가 있으면 먼저 정리
            if (tail_dst.tail >= 0) {
                // clear destination seq_id if it wasn't empty
                llama_kv_cell & cell_dst = cells[tail_dst.tail];

                cell_dst.seq_id.erase(seq_id_dst);
                tail_dst.tail = -1;
                // 셀에 다른 시퀀스가 없으면 완전히 초기화
                if (cell_dst.seq_id.empty()) {
                    cell_dst.pos = -1;
                    cell_dst.delta = -1;
                    cell_dst.src = -1;
                    used -= 1;
                }
            }
            
            // 소스 시퀀스에 상태가 있으면 대상에 복사
            if (tail_src.tail >= 0) {
                llama_kv_cell & cell_src = cells[tail_src.tail];

                cell_src.seq_id.insert(seq_id_dst);
                tail_dst.tail = tail_src.tail;
            }
        }

        return;
    }

    // 트랜스포머 같은 모델의 경우
    head = 0;  // 헤드 위치 초기화

    // 모든 셀을 순회하며 조건에 맞는 셀의 시퀀스 ID 복사
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
            cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

/**
 * 특정 시퀀스만 유지하고 나머지 제거
 * 
 * @param seq_id 유지할 시퀀스 ID
 */
void llama_kv_cache_unified::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        // 순환 모델에서 유지할 seq_id 외의 모든 테일 정보 제거
        if (recurrent && (llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        // 유지할 시퀀스 ID가 없는 셀은 초기화
        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            // 새 헤드 위치 갱신 (가장 먼저 발견된 빈 셀)
            if (new_head == size){
                new_head = i;
            }
        } else {
            // 유지할 시퀀스 ID만 남기고 나머지 제거
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // 빈 슬롯이 생겼고 현재 헤드보다 앞에 있으면 헤드 위치 조정
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

/**
 * 특정 시퀀스의 위치 값을 지정된 범위 내에서 증가/감소시킴
 * 
 * @param seq_id 수정할 시퀀스 ID
 * @param p0     수정 시작 위치 (포함, 기본값 0)
 * @param p1     수정 끝 위치 (제외, 기본값 최대값)
 * @param delta  더할 위치 값 (양수 또는 음수)
 */
void llama_kv_cache_unified::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    // delta가 0이면 작업 불필요
    if (delta == 0) {
        return;
    }

    uint32_t new_head = size;

    // 기본값 설정
    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // 범위가 비어있으면 캐시 순회 없이 빠르게 반환
    if (p0 == p1) {
        return;
    }

    // 순환 모델(RNN) 처리
    if (recurrent) {
        // Mamba나 RWKV 같은 모델은 위치(pos)만 이동시키면 됨
        if (0 <= seq_id && seq_id < (int64_t) size) {
            const int32_t tail_id = cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos += delta;
                }
            }
        }
        return;
    }

    // 트랜스포머 모델 - 모든 셀 순회
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            // 캐시가 시프트 작업이 있었음을 표시
            has_shift = true;
            // 위치와 누적 delta 값 업데이트
            cells[i].pos   += delta;
            cells[i].delta += delta;

            // 위치가 음수가 되면 셀 정리
            if (cells[i].pos < 0) {
                if (!cells[i].is_empty()) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].seq_id.clear();
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // 빈 슬롯이 생겼으면 헤드 위치 조정, 아니면 처음부터 검색
    head = new_head != size ? new_head : 0;
}

/**
 * 특정 시퀀스의 위치 값을 지정된 범위 내에서 나눔
 * 
 * @param seq_id 수정할 시퀀스 ID
 * @param p0     수정 시작 위치 (포함, 기본값 0)
 * @param p1     수정 끝 위치 (제외, 기본값 최대값)
 * @param d      나눌 값 (1이면 작업 수행 안 함)
 */
void llama_kv_cache_unified::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // 1로 나누면 값 변화 없음
    if (d == 1) {
        return;
    }

    // 기본값 설정
    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // 범위가 비어있으면 캐시 순회 없이 빠르게 반환
    if (p0 == p1) {
        return;
    }

    // 순환 모델(RNN) 처리
    if (recurrent) {
        // Mamba나 RWKV 같은 모델은 위치(pos)만 수정하면 됨
        if (0 <= seq_id && seq_id < (int64_t) size) {
            const int32_t tail_id = cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos /= d;
                }
            }
        }

        return;
    }

    // 트랜스포머 모델 - 모든 셀 순회
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            // 캐시가 시프트 작업이 있었음을 표시
            has_shift = true;

            {
                // 원래 위치 저장하고 위치 나누기
                llama_pos p_old = cells[i].pos;
                cells[i].pos   /= d;
                // 변화량을 delta에 누적
                cells[i].delta += cells[i].pos - p_old;
            }
        }
    }
}

/**
 * 특정 시퀀스의 최대 위치 값 반환
 * 
 * @param seq_id 조회할 시퀀스 ID
 * @return       해당 시퀀스의 최대 위치 값
 */
llama_pos llama_kv_cache_unified::seq_pos_max(llama_seq_id seq_id) {
    llama_pos result = 0;

    // 모든 셀 순회하여 최대 위치 찾기
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

/**
 * KV 캐시 조각 모음 요청
 * 다음 추론 시 조각 모음 작업 수행
 */
void llama_kv_cache_unified::defrag() {
    // 순환 모델은 조각 모음 불필요
    if (!recurrent) {
        do_defrag = true;
    }
}

/**
 * 시프트 작업 지원 여부 확인
 * @return 시프트 작업 지원 여부
 */
bool llama_kv_cache_unified::get_can_shift() const {
    return can_shift;
}

/**
 * 배치 처리를 위한 캐시 슬롯 찾기
 * 
 * @param ubatch 처리할 유니파이드 배치(micro-batch) 정보
 * @return       찾은 슬롯 정보 (성공 여부, 시작 위치, 개수 등)
 */
llama_kv_cache_slot_info llama_kv_cache_unified::find_slot(
       const llama_ubatch & ubatch) {
    const uint32_t n_tokens = ubatch.n_tokens;
    const uint32_t n_seqs   = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    if (recurrent) {
        // 순환 모델(Mamba, RWKV 등)의 경우,
        // 각 캐시 셀이 전체 시퀀스 상태를 저장할 수 있음
        // 슬롯은 항상 연속적이어야 함

        // 각 시퀀스에 동일한 수의 새 토큰이 있는 배치만 처리 가능
        GGML_ASSERT(ubatch.equal_seqs);

        int32_t min = size - 1;
        int32_t max = 0;

        // 모든 seq_id가 최대 크기보다 작으면 맞아야 함
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t n_seq_id = ubatch.n_seq_id[s];
            for (uint32_t j = 0; j < n_seq_id; ++j) {
                const llama_seq_id seq_id = ubatch.seq_id[s][j];

                if (seq_id < 0 || (uint32_t) seq_id >= size) {
                    // seq_id가 너무 큼
                    // TODO: 캐시 크기를 조정할 수 있을까?
                    LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%d Try using a bigger --parallel value\n", __func__, seq_id, size);
                    return llama_kv_cache_slot_info_failed;
                }
                if (j > 0) {
                    llama_kv_cell & seq = cells[seq_id];
                    if (seq.tail >= 0) {
                        llama_kv_cell & cell = cells[seq.tail];
                        // 공유되는 seq_id에서 셀을 지움
                        // (일반적으로 발생하지 않지만 처리)
                        cell.seq_id.erase(seq_id);
                        seq.tail = -1;
                        if (cell.seq_id.empty()) {
                            cell.pos = -1;
                            cell.src = -1;
                            used -= 1;
                        }
                    }
                }
            }
        }

#ifndef NDEBUG
        {
            // 디버그 모드에서 테일 검증
            std::vector<int32_t> tails_verif;
            tails_verif.assign(size, -1);
            for (uint32_t i = 0; i < size; ++i) {
                llama_kv_cell & cell = cells[i];
                for (llama_seq_id seq_id : cell.seq_id) {
                    if (tails_verif[seq_id] != -1) {
                        LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                    }
                    tails_verif[seq_id] = i;
                }
            }
            for (uint32_t i = 0; i < size; ++i) {
                if (tails_verif[i] != cells[i].tail) {
                    LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
                }
            }
        }
#endif

        // 다음 빈 셀 찾기
        // 캐시에서 빈 셀을 찾아 새로운 KV 상태를 저장할 위치 탐색
        uint32_t next_empty_cell = head;

        for (uint32_t i = 0; i < size; ++i) {
            // 인덱스가 캐시 크기를 초과하면 순환하여 처음으로 돌아감
            if (next_empty_cell >= size) { next_empty_cell -= size; }
            llama_kv_cell & cell = cells[next_empty_cell];
            // 빈 셀을 찾으면 반복 종료
            if (cell.is_empty()) { break; }
            next_empty_cell += 1;
        }

        // 사용 가능한 셀 범위 찾기
        // 각 시퀀스에 대해 이미 할당된 셀이 있는지 확인하고 없으면 새 셀 할당
        for (uint32_t s = 0; s < n_seqs; ++s) {
            // 시퀀스 ID 가져오기 (배치의 첫 번째 시퀀스 ID)
            const llama_seq_id seq_id = ubatch.seq_id[s][0];
            // 해당 시퀀스 메타데이터 참조 (tail 정보 포함)
            llama_kv_cell & seq_meta = cells[seq_id];
            bool has_cell = false;
            
            // 시퀀스에 이미 할당된 tail 셀이 있는지 확인
            if (seq_meta.tail >= 0) {
                llama_kv_cell & cell = cells[seq_meta.tail];
                GGML_ASSERT(cell.has_seq_id(seq_id));
                // 이 시퀀스 ID가 셀을 "소유"하는지 확인 (셀에 이 시퀀스만 있는지)
                if (cell.seq_id.size() == 1) { has_cell = true; }
            }
            
            // 할당된 셀이 없으면 새 빈 셀 사용
            if (!has_cell) {
                llama_kv_cell & empty_cell = cells[next_empty_cell];
                GGML_ASSERT(empty_cell.is_empty());
                
                // 기존 tail이 있으면 빈 셀로 정보 복사
                if (seq_meta.tail >= 0) {
                    llama_kv_cell & orig_cell = cells[seq_meta.tail];
                    empty_cell.pos = orig_cell.pos;  // 위치 복사
                    empty_cell.src = orig_cell.src;  // 소스 복사
                    orig_cell.seq_id.erase(seq_id);  // 원본 셀에서 시퀀스 ID 제거
                    empty_cell.seq_id.insert(seq_id); // 새 셀에 시퀀스 ID 추가 (나중에 덮어씀)
                }
                
                // 시퀀스 메타데이터의 tail 업데이트
                seq_meta.tail = next_empty_cell;
                
                // 다음 시퀀스를 위한 빈 셀 찾기
                if (s + 1 < n_seqs) {
                    next_empty_cell += 1;
                    // 다음 빈 셀 탐색
                    for (uint32_t i = 0; i < size; ++i) {
                        if (next_empty_cell >= size) { next_empty_cell -= size; }
                        llama_kv_cell & cell = cells[next_empty_cell];
                        if (cell.is_empty()) { break; }
                        next_empty_cell += 1;
                    }
                }
            }
            
            // 연속적인 셀 범위의 최소값과 최대값 업데이트
            if (min > seq_meta.tail) { min = seq_meta.tail; }
            if (max < seq_meta.tail) { max = seq_meta.tail; }
        }

        // 셀 모으기 및 재정렬
        // 셀들을 연속적인 메모리 범위로 모으기 위해 셀 교환
        for (uint32_t s = 0; s < n_seqs; ++s) {
            // 대상 위치와 현재 소스 위치 계산
            int32_t dst_id = s + min;  // 연속적인 범위의 시작점부터의 오프셋
            int32_t src_id = cells[ubatch.seq_id[s][0]].tail;  // 시퀀스의 현재 tail 위치
            
            // 위치가 다르면 셀 데이터 교환
            if (dst_id != src_id) {
                llama_kv_cell & dst_cell = cells[dst_id];
                llama_kv_cell & src_cell = cells[src_id];

                // 셀 데이터 교환 (pos, src, seq_id)
                std::swap(dst_cell.pos, src_cell.pos);
                std::swap(dst_cell.src, src_cell.src);
                std::swap(dst_cell.seq_id, src_cell.seq_id);

                // 시퀀스의 tail 포인터 업데이트 (서로 절대 겹치지 않는다고 가정)
                // 소스 셀에 있던 시퀀스들의 tail을 src_id로 업데이트
                for (const llama_seq_id seq_id : src_cell.seq_id) {
                    cells[seq_id].tail = src_id;
                }
                // 대상 셀에 있던 시퀀스들의 tail을 dst_id로 업데이트
                for (const llama_seq_id seq_id : dst_cell.seq_id) {
                    cells[seq_id].tail = dst_id;
                }
            }
        }

        // 사용된 시퀀스의 pos 업데이트
        // 새 토큰으로 셀의 위치 정보 갱신
        for (uint32_t s = 0; s < n_seqs; ++s) {
            // 해당 시퀀스의 마지막 토큰 위치 가져오기
            const llama_pos last_pos = ubatch.pos[n_seq_tokens * s + n_seq_tokens - 1];
            int32_t cell_id = s + min;  // 셀 ID 계산
            llama_kv_cell & cell = cells[cell_id];

            // 위치 일관성 검사: 연속적이지 않은 위치 값 경고
            if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
                // 위치가 역행하거나 값을 건너뛸 때 어떻게 해야 할까?
                // 배치 중간에 상태를 지우려면 특별한 처리가 필요한데 현재 구현되지 않음
                LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                    __func__, last_pos, cell.pos, ubatch.seq_id[s][0], n_seq_tokens);
            }
            
            // 셀 위치 업데이트 및 시퀀스 ID 재설정
            cell.pos = last_pos;
            cell.seq_id.clear();
            
            // 배치의 모든 시퀀스 ID를 셀에 추가하고 해당 시퀀스의 tail 업데이트
            for (int32_t j = 0; j < ubatch.n_seq_id[s]; ++j) {
                const llama_seq_id seq_id = ubatch.seq_id[s][j];
                cell.seq_id.insert(seq_id);
                cells[seq_id].tail = cell_id;
            }
        }

        // 사용된 셀 범위 정보 업데이트 (head부터 head + n까지)
        head = min;  // 헤드 위치를 연속 범위의 시작으로 설정
        n    = max - min + 1;  // 셀 개수 계산
        
        // 사용 중인 셀 수 다시 계산 (비어 있지 않은 셀 개수)
        used = std::count_if(cells.begin(), cells.end(),
            [](const llama_kv_cell& cell){ return !cell.is_empty(); });

        // 무결성 검사: 사용된 셀 수가 시퀀스 수 이상인지 확인
        return llama_kv_cache_slot_info(n >= n_seqs);
    }

    // 트랜스포머 모델용: 토큰당 하나의 셀 사용
    // 순환 모델과 달리 각 토큰이 별도의 캐시 셀을 필요로 함

    // 필요한 토큰 수가 캐시 크기보다 크면 오류 반환
    if (n_tokens > size) {
        LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %d\n", __func__, n_tokens, size);
        return llama_kv_cache_slot_info_failed;
    }

    uint32_t n_tested = 0;  // 검사한 셀의 수

    // 연속된 빈 셀 찾기
    while (true) {
        // 현재 헤드 위치에서 필요한 토큰 수만큼의 공간이 없으면 
        // 처음부터 다시 검색
        if (head + n_tokens > size) {
            n_tested += size - head;  // 검사한 셀 수 갱신
            head = 0;  // 헤드를 처음으로 리셋
            continue;
        }

        // 연속된 빈 셀 검사
        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            // 비어 있지 않은 셀이 있으면
            if (cells[head + i].pos >= 0) {
                found = false;
                head     += i + 1;  // 헤드 위치 이동
                n_tested += i + 1;  // 검사한 셀 수 갱신
                break;
            }
        }

        // 연속된 빈 셀을 찾았으면 반복 종료
        if (found) {
            break;
        }

        // 모든 셀을 검사했는데도 충분한 공간이 없으면 실패 반환
        if (n_tested >= size) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return llama_kv_cache_slot_info_failed;
        }
    }

    // 찾은 슬롯에 배치의 토큰 정보 저장
    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            uint32_t k = s*n_seq_tokens + i;  // 토큰의 전체 인덱스
            cells[head + k].pos = ubatch.pos[k];  // 토큰 위치 저장

            // 해당 시퀀스의 모든 시퀀스 ID를 셀에 추가
            for (int32_t j = 0; j < ubatch.n_seq_id[s]; j++) {
                cells[head + k].seq_id.insert(ubatch.seq_id[s][j]);
            }
        }
    }

    // 사용 중인 셀 수 증가
    used += n_tokens;

    // 슬롯 정보 반환 (시작과 끝 위치)
    return llama_kv_cache_slot_info(head, head + n_tokens);
}

/**
 * Flash Attention 커널에 필요한 패딩 크기 반환
 * 
 * @param cparams 컨텍스트 파라미터
 * @return        패딩 크기 (바이트)
 */
uint32_t llama_kv_cache_unified::get_padding(const llama_cparams & cparams) const {
    // Flash Attention 커널은 런타임 경계 검사를 피하기 위해 추가 패딩 필요
    return cparams.flash_attn ? 256u : 32u;
}

/**
 * 캐시에서 사용된 최대 셀 인덱스 반환
 * 
 * @return 캐시에서 사용된 가장 큰 셀 인덱스 (비어있으면 0)
 */
uint32_t llama_kv_cache_unified::cell_max() const {
    // 뒤에서부터 검색하여 비어있지 않은 첫 번째 셀 인덱스 반환
    for (uint32_t i = size; i > 0; --i) {
        const llama_kv_cell & cell = cells[i - 1];

        // 유효한 위치를 가지고 비어있지 않은 셀 찾기
        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    // 모든 셀이 비어있음
    return 0;
}

/**
 * 모든 K(key) 텐서의 총 메모리 크기 계산
 * 
 * @return K 텐서가 사용하는 총 메모리 크기 (바이트)
 */
size_t llama_kv_cache_unified::size_k_bytes() const {
    size_t size_k_bytes = 0;

    // 모든 레이어의 K 텐서 크기 합산
    for (const auto & k : k_l) {
        size_k_bytes += ggml_nbytes(k);
    }

    return size_k_bytes;
}

/**
 * 모든 V(value) 텐서의 총 메모리 크기 계산
 * 
 * @return V 텐서가 사용하는 총 메모리 크기 (바이트)
 */
size_t llama_kv_cache_unified::size_v_bytes() const {
    size_t size_v_bytes = 0;

    // 모든 레이어의 V 텐서 크기 합산
    for (const auto & v : v_l) {
        size_v_bytes += ggml_nbytes(v);
    }

    return size_v_bytes;
}

/**
 * KV 캐시 조각 모음(defragmentation) 준비
 * 비어있는 셀(hole)을 찾고 캐시 끝에서부터 사용 중인 셀을 이 빈 공간으로 이동하는 계획을 세움
 * 
 * @param n_max_nodes 최대 계산 그래프 노드 수 제한
 * @return           조각 모음 준비 성공 여부
 */
bool llama_kv_cache_unified::defrag_prepare(int32_t n_max_nodes) {
    // 시간 측정 시작
    const int64_t t_start = ggml_time_us();

    // 모델의 레이어 수 가져오기
    const uint32_t n_layer = hparams.n_layer;

    // 전체 캐시 셀 개수와 실제 사용 중인 셀 개수
    const uint32_t n_kv   = cell_max();  // 사용된 가장 큰 셀 인덱스 (실질적인 캐시 크기)
    const uint32_t n_used = used;        // 실제로 사용 중인 셀의 수

    // 사용 중인 셀 수가 총 셀 수보다 크면 안 됨 (안전성 검사)
    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // 이동된 셀 수를 추적
    uint32_t n_moves = 0;

    // 각 셀 이동은 6*n_layer 텐서가 필요 (graph_build_kv_self_defrag 참조)
    //   - 소스 뷰, 대상 뷰, 복사 연산
    //   - 키와 값에 각각 적용 (×2)
    //const uint32_t max_moves = max_nodes()/(6*n_layer);
    // TODO: 임시 수정 - https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    // 최대 이동 가능 셀 수 계산 (노드 제한 고려)
    // 셀 이동은 GGML 계산 그래프에서 노드를 소비하므로 최대 노드 수를 기반으로 제한함
    const uint32_t max_moves = (n_max_nodes - 2*n_layer)/(6*n_layer);

    // 어떤 KV 셀을 어디로 이동할지 결정
    //
    // ids[i] 값이 셀 i의 이동 목적지 위치를 나타냄
    // 예: ids[10] = 5는 10번 셀을 5번 위치로 이동한다는 의미
    //
    // ids[i] == i 또는 ids[i] == n_kv인 경우 셀 i는 이동하지 않음
    // (자기 자신의 위치이거나 기본값인 경우)
    //
    auto & ids = defrag_info.ids;

    // ids 배열 초기화 (기본값 n_kv로 설정 = 이동하지 않음)
    ids.clear();
    ids.resize(n_kv, n_kv);

    // 사용 중인 셀들을 조사하며 캐시 최적화 계획 수립
    // i0는 검사 중인 셀의 인덱스 (앞에서부터 검색)
    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        const auto & cell0 = cells[i0];

        // 셀이 비어있지 않으면 현재 위치 유지 (이동 불필요)
        if (!cell0.is_empty()) {
            ids[i0] = i0;  // 자기 자신 위치로 설정 (= 이동 안 함)
            continue;
        }

        // 빈 공간(홀) 발견 - 캐시 끝에서 데이터를 가져와 채움

        uint32_t nh = 1;  // 현재 홀(hole)의 크기 (연속된 빈 셀 개수)

        // 연속된 빈 공간(홀)의 크기 계산
        // 현재 셀 이후로 연속된 빈 셀 개수 세기
        while (i0 + nh < n_used && cells[i0 + nh].is_empty()) {
            nh++;
        }

        uint32_t nf = 0;  // 찾은 채울 셀 수 (found cells)
        uint32_t is = n_kv - 1;  // 캐시 끝에서부터 검색 시작 위치

        // 끝에서부터 nh개의 비어있지 않은 셀 찾기
        // 캐시 뒷부분의 사용 중인 셀을 앞쪽 빈 공간으로 옮기는 전략
        // 이는 메모리 지역성(locality)을 개선하고 캐시 효율성을 높임
        for (; is > i0; --is) {
            const auto & cell1 = cells[is];

            // 빈 셀이거나 이미 이동 계획된 셀은 건너뜀
            if (cell1.is_empty() || ids[is] != n_kv) {
                continue;
            }

            // 비어있지 않고 아직 이동 계획되지 않은 셀 찾음
            nf++;

            // 필요한 빈 공간 크기만큼 채울 셀을 찾으면 중단
            if (nf == nh) {
                break;
            }
        }

        // nf가 nh와 같지 않은 경우는 n_used가 정확하지 않을 때만 발생 (버그)
        // 모든 홀을 채울 충분한 사용 중인 셀을 찾아야 함
        // 이 검증은 알고리즘의 정확성을 보장함
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;  // 찾은 셀 카운터 초기화

        uint32_t i1 = is;  // 이동 대상 셀 인덱스 시작 (뒤에서부터)

        // 연속된 메모리 블록을 이동하는지 여부
        // 셀 이동 시 연속된 메모리 블록은 단일 이동으로 처리하여 효율성 증가
        bool cont = false;

        // 다음 이동 검색을 중단해야 하는지 여부
        bool stop = false;

        // 뒤에서부터 찾은 셀들을 빈 공간으로 이동
        for (; i1 < n_kv; ++i1) {
            auto & cell1 = cells[i1];

            // 빈 셀이거나 이미 이동 계획된 셀은 건너뜀
            if (cell1.is_empty() || ids[i1] != n_kv) {
                // 최대 이동 횟수에 도달하면 검색 중단
                // 한 번의 조각 모음에서 너무 많은 이동이 발생하면 효율성 저하
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;  // 연속성 깨짐
                continue;
            }

            // 셀 i1은 (i0 + nf) 위치로 이동할 계획
            ids[i1] = i0 + nf;  // 이동 목적지 기록

            // 셀 메타데이터 이동
            // 실제 텐서 데이터는 아직 이동하지 않고, 메타데이터만 미리 복사
            // 실제 데이터 이동은 defrag() 함수에서 이 계획을 바탕으로 수행됨
            cells[i0 + nf] = cell1;

            // 원래 셀 비우고 헤드 위치 업데이트
            cell1 = llama_kv_cell();  // 원본 셀 초기화
            head = n_used;  // 헤드 위치 업데이트 (다음 셀 삽입 위치)

            // 연속된 메모리 이동 여부 추적
            // 연속된 셀들은 하나의 이동으로 간주하여 이동 카운트 최적화
            // 이는 GGML 그래프에서 더 효율적인 텐서 연산으로 변환됨
            if (!cont) {
                n_moves++;  // 새로운 이동 시작
                cont = true;  // 연속 플래그 활성화
            }

            nf++;  // 이동한 셀 개수 증가

            // 필요한 빈 공간을 모두 채웠으면 중단
            if (nf == nh) {
                break;
            }
        }

        // 최대 이동 횟수에 도달하면 검색 중단
        // 부분적인 조각 모음만 수행하고 다음 호출에서 계속 진행
        if (stop || n_moves == max_moves) {
            break;
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        // 다음 검사 위치 업데이트 (처리한 홀 크기만큼 건너뜀)
        i0 += nh - 1;  // -1은 다음 for 반복에서 +1이 되기 때문
    }

    // 이동할 셀이 없으면 조각 모음 불필요
    // 실질적으로 최적화할 것이 없는 경우 (빈 공간이 없거나 모든 셀이 이미 최적화됨)
    if (n_moves == 0) {
        return false;
    }

    // 시간 측정 종료
    const int64_t t_end = ggml_time_us();
    
    // 디버그 로그: 이동 계획된 셀 수 출력
    LLAMA_LOG_DEBUG("(tmp log) KV defrag cell moves: %u\n", n_moves);

    // 예상되는 GGML 계산 그래프 노드 수 출력
    // 이는 계산 자원 사용량 예측에 사용됨
    LLAMA_LOG_DEBUG("expected gf nodes: %u\n", 6*n_moves*n_layer);
    
    // defrag_prepare 함수의 실행 시간 출력
    LLAMA_LOG_INFO("%s: defrag_prepare took %.3f ms\n", __func__, (t_end - t_start) / 1000.0f);

    // 조각 모음 계획 수립 완료
    // 실제 셀 데이터 이동은 defrag() 함수에서 수행됨
    return true;
}

/**
 * KV 캐시 상태 파일에 저장
 * 특정 시퀀스 또는 전체 캐시의 상태를 출력 스트림에 기록
 * 
 * @param io      출력 스트림 인터페이스
 * @param seq_id  저장할 시퀀스 ID (기본값 -1은 모든 시퀀스 저장)
 */
void llama_kv_cache_unified::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    // 저장할 셀 범위 저장 (시작 인덱스 포함, 끝 인덱스 제외)
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; 
    uint32_t cell_count = 0;

    // 지정된 seq_id가 있는 셀의 수 계산 및 연속된 범위 찾기
    // 시퀀스 ID가 -1이면 모든 비어있지 않은 셀 포함
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            // 새 범위 시작점 기록
            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            // 현재 범위 종료
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    // 마지막 범위 처리
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    // 디버그 검사: 범위 내 셀 합계가 총 셀 수와 일치하는지 확인
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    // 총 셀 수 저장
    io.write(&cell_count, sizeof(cell_count));

    // 메타데이터 저장 (위치, 시퀀스 ID 등)
    state_write_meta(io, cell_ranges, seq_id);
    // 텐서 데이터 저장 (K, V 텐서)
    state_write_data(io, cell_ranges);
}

/**
 * KV 캐시 상태 파일에서 읽기
 * 입력 스트림에서 KV 캐시 상태를 복원
 * 
 * @param io      입력 스트림 인터페이스
 * @param seq_id  복원할 시퀀스 ID (기본값 -1은 전체 캐시 복원)
 */
void llama_kv_cache_unified::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    // 셀 수 읽기
    uint32_t cell_count;
    io.read_to(&cell_count, sizeof(cell_count));

    bool res = true;
    // 메타데이터 읽기 (위치, 시퀀스 ID 등)
    res = res && state_read_meta(io, cell_count, seq_id);
    // 텐서 데이터 읽기 (K, V 텐서)
    res = res && state_read_data(io, cell_count);

    // 읽기 실패 시 처리
    if (!res) {
        if (seq_id == -1) {
            // 전체 캐시를 복원하려 했으면 모두 초기화
            clear();
        } else {
            // 특정 시퀀스만 복원하려 했으면 해당 시퀀스만 제거
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }
}

/**
 * KV 캐시 메타데이터 저장
 * 셀 위치, 시퀀스 ID 등의 메타데이터를 출력 스트림에 기록
 * 
 * @param io          출력 스트림 인터페이스
 * @param cell_ranges 저장할 셀 범위 목록 (각 쌍은 시작과 끝 인덱스)
 * @param seq_id      저장할 시퀀스 ID (-1은 모든 시퀀스)
 */
void llama_kv_cache_unified::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    // 모든 셀 범위를 순회하며 메타데이터 저장
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;  // 셀 위치
            // 시퀀스 ID 개수 (특정 시퀀스만 저장하는 경우 0)
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            // 위치와 시퀀스 ID 개수 저장
            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            // 모든 시퀀스를 저장하는 경우 각 시퀀스 ID 저장
            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

/**
 * KV 캐시 텐서 데이터 저장
 * K, V 텐서 데이터를 출력 스트림에 기록
 * 
 * @param io          출력 스트림 인터페이스
 * @param cell_ranges 저장할 셀 범위 목록
 */
void llama_kv_cache_unified::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    // V 텐서 전치 여부와 레이어 수 저장
    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = hparams.n_layer;

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;

    // 모든 레이어의 K(key) 텐서 저장
    // 각 셀에 대해 한 행씩 저장
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // K 텐서 타입 저장
        const int32_t k_type_i = (int32_t)k_l[il]->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // K 텐서 행 크기 저장
        const uint64_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // 각 셀 범위의 K 텐서 데이터 저장
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k_l[il], range.first * k_size_row, buf_size);
        }
    }

    // V 텐서가 전치되지 않은 경우 (일반적인 형태)
    if (!v_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // V 텐서 타입 저장
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // V 텐서 행 크기 저장
            const uint64_t v_size_row = ggml_row_size(v_l[il]->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // 각 셀 범위의 V 텐서 데이터 저장
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v_l[il], range.first * v_size_row, buf_size);
            }
        }
    } else {
        // V 텐서가 전치된 경우 (각 요소 크기와 임베딩 차원도 저장)
        const uint32_t kv_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // V 텐서 타입 저장
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // V 텐서 요소 크기 저장
            const uint32_t v_size_el = ggml_type_size(v_l[il]->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // GQA 임베딩 크기 저장
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // 각 임베딩 차원에 대해 모든 셀의 요소 저장
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

/**
 * KV 캐시 메타데이터 복원
 * 입력 스트림에서 메타데이터를 읽어 캐시 상태 복원
 * 
 * @param io          입력 스트림 인터페이스
 * @param cell_count  읽을 셀 개수
 * @param dest_seq_id 대상 시퀀스 ID (특정 시퀀스로 복원 시 사용, -1은 원본 그대로 복원)
 * @return            메타데이터 복원 성공 여부
 */
bool llama_kv_cache_unified::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // 단일 시퀀스로 복원하는 경우

        // 기존 시퀀스 제거
        seq_rm(dest_seq_id, -1, -1);

        // 배치 준비 (셀을 할당하기 위해)
        llama_sbatch sbatch;
        llama_ubatch batch = sbatch.reserve_ubatch(cell_count, /* has_embd */ false);

        batch.n_tokens = cell_count;
        batch.n_seq_tokens = cell_count;
        batch.n_seqs = 1;

        // 각 셀의 위치 정보 읽기
        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            // 시퀀스 ID 무관 셀이 아니면 오류
            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            batch.pos[i] = pos;
        }
        // 배치에 대상 시퀀스 ID 설정
        batch.n_seq_id[0] = 1;
        batch.seq_id[0] = &dest_seq_id;
        
        // 사용 가능한 셀 슬롯 찾기
        if (!find_slot(batch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // 디버그 검사: 첫 셀이 head, 마지막 셀이 head+cell_count-1인지 확인
        // 셀들이 연속된 블록을 이루는지 확인
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == batch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == batch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // 전체 KV 캐시 복원

        // 셀 개수 검사
        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // 캐시 초기화
        clear();

        // 셀 메타데이터 복원
        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_kv_cell & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            // 위치와 시퀀스 ID 개수 읽기
            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            // 각 시퀀스 ID 복원
            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                // 시퀀스 ID 유효성 검사
                // TODO: llama_kv_cache_unified should have a notion of max sequences
                //if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(ctx)) {
                if (seq_id < 0) {
                    //LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, llama_n_seq_max(ctx));
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, inf)\n", __func__, seq_id);
                    return false;
                }

                // 시퀀스 ID 추가
                cell.seq_id.insert(seq_id);

                // 순환 모델인 경우 tail 포인터 설정
                if (recurrent) {
                    int32_t & tail = cells[seq_id].tail;
                    if (tail != -1) {
                        LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                        return false;
                    }
                    tail = i;
                }
            }
        }

        // 캐시 상태 설정
        head = 0;
        used = cell_count;
    }

    // 순환 모델인 경우 소스 셀 ID 설정 (상태 유지를 위해)
    if (recurrent) {
        for (uint32_t i = 0; i < cell_count; ++i) {
            uint32_t cell_id = head + i;
            // 복원된 순환 상태가 유지되도록 소스 셀 ID 설정
            cells[cell_id].src = cell_id;
        }
    }

    return true;
}

/**
 * KV 캐시 텐서 데이터 복원
 * 입력 스트림에서 K, V 텐서 데이터를 읽어서 캐시에 로드
 * 
 * @param io          입력 스트림 인터페이스
 * @param cell_count  읽을 셀 개수
 * @return            데이터 복원 성공 여부
 */
bool llama_kv_cache_unified::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    // V 텐서 전치 여부와 레이어 수 읽기
    uint32_t v_trans;
    uint32_t n_layer;
    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    // 레이어 수 일치 여부 확인
    if (n_layer != hparams.n_layer) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer);
        return false;
    }
    // 셀 개수가 캐시 크기보다 작은지 확인
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    // V 텐서 전치 여부 호환성 확인
    if (v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // 각 레이어별로 K(key) 텐서 데이터 읽기
    // 한 행이 한 셀에 해당, 연속된 블록으로 읽음
    for (uint32_t il = 0; il < n_layer; ++il) {
        // GQA(Grouped-Query Attention)를 위한 임베딩 차원 계산
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // K 텐서 타입 읽고 일치 여부 확인
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k_l[il]->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // K 텐서 행 크기 읽고 일치 여부 확인
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // 전체 셀 범위에 대한 K 텐서 데이터 읽고 설정
            ggml_backend_tensor_set(k_l[il], io.read(cell_count * k_size_row), head * k_size_row, cell_count * k_size_row);
        }
    }

    // V 텐서가 전치되지 않은 경우 (일반적인 형태)
    if (!v_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // GQA를 위한 임베딩 차원 계산
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // V 텐서 타입 읽고 일치 여부 확인
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // V 텐서 행 크기 읽고 일치 여부 확인
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v_l[il]->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // 전체 셀 범위에 대한 V 텐서 데이터 읽고 설정
                ggml_backend_tensor_set(v_l[il], io.read(cell_count * v_size_row), head * v_size_row, cell_count * v_size_row);
            }
        }
    } else {
        // V 텐서가 전치된 경우 (각 열이 같은 임베딩 차원의 모든 셀을 포함)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // GQA를 위한 임베딩 차원 계산
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // V 텐서 타입 읽고 일치 여부 확인
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // V 텐서 요소 크기 읽고 일치 여부 확인
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v_l[il]->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // GQA 임베딩 크기 읽고 일치 여부 확인
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                // 전치된 행렬의 각 행에 대해 전체 셀 범위의 V 텐서 데이터 읽기
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    // 대상 오프셋 계산: (head + j * size) * v_size_el
                    // 전치된 형태에서는 같은 임베딩 차원 j의 모든 셀이 연속되어 있음
                    const size_t dst_offset = (head + j * size) * v_size_el;
                    ggml_backend_tensor_set(v_l[il], io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                }
            }
        }
    }

    return true;
}

//
// interface implementation
//

/**
 * KV 캐시에 저장된 토큰 수 반환
 * 
 * @param kv KV 캐시 포인터
 * @return   저장된 토큰 수 (캐시가 없으면 0)
 */
int32_t llama_kv_cache_n_tokens(const llama_kv_cache * kv) {
    if (!kv) {
        return 0;
    }

    return kv->get_n_tokens();
}

/**
 * KV 캐시에서 사용 중인 셀 수 반환
 * 
 * @param kv KV 캐시 포인터
 * @return   사용 중인 셀 수 (캐시가 없으면 0)
 */
int32_t llama_kv_cache_used_cells(const llama_kv_cache * kv) {
    if (!kv) {
        return 0;
    }

    return kv->get_used_cells();
}

/**
 * KV 캐시 전체 초기화
 * 모든 셀과 상태를 리셋
 * 
 * @param kv KV 캐시 포인터
 */
void llama_kv_cache_clear(llama_kv_cache * kv) {
    if (!kv) {
        return;
    }

    kv->clear();
}

/**
 * 특정 시퀀스 ID의 특정 범위 위치 토큰 제거
 * 
 * @param kv     KV 캐시 포인터
 * @param seq_id 제거할 시퀀스 ID
 * @param p0     제거 시작 위치 (포함, 기본값 0)
 * @param p1     제거 끝 위치 (제외, 기본값 최대값)
 * @return       제거 성공 여부 (캐시가 없으면 true)
 */
bool llama_kv_cache_seq_rm(
        llama_kv_cache * kv,
          llama_seq_id   seq_id,
             llama_pos   p0,
             llama_pos   p1) {
    if (!kv) {
        return true;
    }

    return kv->seq_rm(seq_id, p0, p1);
}

/**
 * 한 시퀀스의 상태를 다른 시퀀스로 복사
 * 
 * @param kv         KV 캐시 포인터
 * @param seq_id_src 소스 시퀀스 ID
 * @param seq_id_dst 대상 시퀀스 ID
 * @param p0         복사 시작 위치 (포함, 기본값 0)
 * @param p1         복사 끝 위치 (제외, 기본값 최대값)
 */
void llama_kv_cache_seq_cp(
        llama_kv_cache * kv,
          llama_seq_id   seq_id_src,
          llama_seq_id   seq_id_dst,
             llama_pos   p0,
             llama_pos   p1) {
    if (!kv) {
        return;
    }

    kv->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

/**
 * 특정 시퀀스만 유지하고 나머지 제거
 * 
 * @param kv     KV 캐시 포인터
 * @param seq_id 유지할 시퀀스 ID
 */
void llama_kv_cache_seq_keep(llama_kv_cache * kv, llama_seq_id seq_id) {
    if (!kv) {
        return;
    }

    kv->seq_keep(seq_id);
}

/**
 * 특정 시퀀스의 위치 값을 지정된 범위 내에서 증가/감소시킴
 * 
 * @param kv     KV 캐시 포인터
 * @param seq_id 수정할 시퀀스 ID
 * @param p0     수정 시작 위치 (포함, 기본값 0)
 * @param p1     수정 끝 위치 (제외, 기본값 최대값)
 * @param delta  더할 위치 값 (양수 또는 음수)
 */
void llama_kv_cache_seq_add(
        llama_kv_cache * kv,
          llama_seq_id   seq_id,
             llama_pos   p0,
             llama_pos   p1,
             llama_pos   delta) {
    if (!kv) {
        return;
    }

    kv->seq_add(seq_id, p0, p1, delta);
}

/**
 * 특정 시퀀스의 위치 값을 지정된 범위 내에서 나눔
 * 
 * @param kv     KV 캐시 포인터
 * @param seq_id 수정할 시퀀스 ID
 * @param p0     수정 시작 위치 (포함, 기본값 0)
 * @param p1     수정 끝 위치 (제외, 기본값 최대값)
 * @param d      나눌 값 (1이면 작업 수행 안 함)
 */
void llama_kv_cache_seq_div(
        llama_kv_cache * kv,
          llama_seq_id   seq_id,
             llama_pos   p0,
             llama_pos   p1,
                   int   d) {
    if (!kv) {
        return;
    }

    kv->seq_div(seq_id, p0, p1, d);
}

/**
 * 특정 시퀀스의 최대 위치 값 반환
 * 
 * @param kv     KV 캐시 포인터
 * @param seq_id 조회할 시퀀스 ID
 * @return       해당 시퀀스의 최대 위치 값 (캐시가 없으면 0)
 */
llama_pos llama_kv_cache_seq_pos_max(llama_kv_cache * kv, llama_seq_id seq_id) {
    if (!kv) {
        return 0;
    }

    return kv->seq_pos_max(seq_id);
}

/**
 * KV 캐시 조각 모음 요청
 * 다음 추론 시 조각 모음 작업 수행
 * 
 * @param kv KV 캐시 포인터
 */
void llama_kv_cache_defrag(llama_kv_cache * kv) {
    if (!kv) {
        return;
    }

    kv->defrag();
}

/**
 * 시프트 작업 지원 여부 확인
 * 
 * @param kv KV 캐시 포인터
 * @return   시프트 작업 지원 여부 (캐시가 없으면 false)
 */
bool llama_kv_cache_can_shift(const llama_kv_cache * kv) {
    if (!kv) {
        return false;
    }

    return kv->get_can_shift();
}

//
// KV 캐시 뷰 (kv cache view)
//

/**
 * KV 캐시 뷰 초기화
 * 캐시 상태를 확인하기 위한 뷰 객체 생성
 * 
 * @param kv        KV 캐시 참조
 * @param n_seq_max 최대 시퀀스 수
 * @return          초기화된 KV 캐시 뷰
 */
llama_kv_cache_view llama_kv_cache_view_init(const llama_kv_cache & kv, int32_t n_seq_max) {
    llama_kv_cache_view result = {
        /*.n_cells            = */ 0,
        /*.n_seq_max          = */ n_seq_max,
        /*.token_count        = */ 0,
        /*.used_cells         = */ llama_kv_cache_used_cells(&kv),
        /*.max_contiguous     = */ 0,
        /*.max_contiguous_idx = */ -1,
        /*.cells              = */ nullptr,
        /*.cells_sequences    = */ nullptr,
    };

    return result;
}

/**
 * KV 캐시 뷰 해제
 * 뷰 객체에 할당된 메모리 정리
 * 
 * @param view 해제할 KV 캐시 뷰 포인터
 */
void llama_kv_cache_view_free(llama_kv_cache_view * view) {
    if (view->cells != nullptr) {
        free(view->cells);
        view->cells = nullptr;
    }
    if (view->cells_sequences != nullptr) {
        free(view->cells_sequences);
        view->cells_sequences = nullptr;
    }
}

/**
 * KV 캐시 뷰 업데이트
 * 현재 캐시 상태로 뷰 정보 갱신
 * 
 * @param view KV 캐시 뷰 포인터
 * @param kv   KV 캐시 포인터
 */
void llama_kv_cache_view_update(llama_kv_cache_view * view, const llama_kv_cache * kv) {
    // TODO: 향후 개선 필요, 현재는 빠른 임시 방법
    const llama_kv_cache_unified * kvu = dynamic_cast<const llama_kv_cache_unified *>(kv);
    if (kvu == nullptr) {
        LLAMA_LOG_ERROR("%s: the kv_cache_view currently works only with llama_kv_cache_unified\n", __func__);
        return;
    }

    // 필요한 경우 뷰 메모리 재할당
    if (uint32_t(view->n_cells) < kvu->size || view->cells == nullptr) {
        view->n_cells = int32_t(kvu->size);
        // 셀 정보를 위한 메모리 할당
        void * p = realloc(view->cells, sizeof(llama_kv_cache_view_cell) * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells");
        view->cells = (llama_kv_cache_view_cell *)p;
        // 셀 시퀀스 정보를 위한 메모리 할당
        p = realloc(view->cells_sequences, sizeof(llama_seq_id) * view->n_seq_max * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells sequences");
        view->cells_sequences = (llama_seq_id *)p;
    }

    // 캐시 셀 정보 복사 및 통계 계산
    const std::vector<llama_kv_cell> & kv_cells = kvu->cells;
    llama_kv_cache_view_cell * c_curr = view->cells;
    llama_seq_id * cs_curr = view->cells_sequences;
    int32_t used_cells = 0;
    int32_t token_count = 0;
    int32_t curr_contig_idx = -1;
    uint32_t max_contig = 0;
    int32_t max_contig_idx = -1;

    for (int32_t i = 0; i < int32_t(kvu->size); i++, c_curr++, cs_curr += view->n_seq_max) {
        // 현재 셀의 시퀀스 ID 개수
        const size_t curr_size = kv_cells[i].seq_id.size();
        // 토큰 수 집계
        token_count += curr_size;
        // delta를 고려한 실제 위치 값 계산
        c_curr->pos = kv_cells[i].pos + kv_cells[i].delta;

        // 연속된 빈 셀 영역 탐지 및 최대 영역 갱신
        if (curr_size > 0) {
            // 비어 있지 않은 셀을 만나면 이전까지의 연속 빈 영역 확인
            if (curr_contig_idx >= 0 && uint32_t(i - curr_contig_idx) > max_contig) {
                max_contig = i - curr_contig_idx;
                max_contig_idx = curr_contig_idx;
            }
            // 연속성 초기화 (비어 있지 않은 셀 발견)
            curr_contig_idx = -1;
        } else if (curr_contig_idx < 0) {
            // 빈 셀 시작점 기록
            curr_contig_idx = i;
        }

        // 현재 셀의 시퀀스 ID 정보 복사
        int seq_idx = 0;
        for (const llama_seq_id it : kv_cells[i].seq_id) {
            // 최대 시퀀스 수 제한 확인
            if (seq_idx >= view->n_seq_max) {
                break;
            }
            // 시퀀스 ID 복사
            cs_curr[seq_idx] = it;
            seq_idx++;
        }
        // 셀에 시퀀스 ID가 있으면 사용 중인 셀로 카운트
        if (seq_idx != 0) {
            used_cells++;
        }
        // 남은 시퀀스 ID 슬롯을 -1로 초기화(사용하지 않는 슬롯 표시)
        for (; seq_idx < view->n_seq_max; seq_idx++) {
            cs_curr[seq_idx] = -1;
        }
    }
    
    // 마지막 연속 빈 셀 영역 처리
    if (curr_contig_idx >= 0 && kv_cells.size() - curr_contig_idx > max_contig) {
        max_contig_idx = curr_contig_idx;
        max_contig = kv_cells.size() - curr_contig_idx;
    }
    
    // 뷰 정보 갱신
    view->max_contiguous = max_contig;       // 최대 연속 빈 공간 크기
    view->max_contiguous_idx = max_contig_idx; // 최대 연속 빈 공간 시작 인덱스
    view->token_count = token_count;         // 총 토큰 수
    view->used_cells = used_cells;           // 사용 중인 셀 수
    
    // 일관성 검사: 계산된 사용 셀 수와 캐시 내부 상태 비교
    if (uint32_t(used_cells) != kvu->used) {
        LLAMA_LOG_ERROR("%s: used cells mismatch. kv_cache says %d but we calculated %d\n",
            __func__, kvu->used, used_cells);
    }
}
