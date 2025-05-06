// llama-context.cpp - LLaMA 모델의 추론 컨텍스트 관리를 위한 코드
// 이 파일은 모델 상태, KV 캐시, 계산 그래프 등을 관리하는 핵심 컴포넌트입니다

#include "llama-context.h"  // 컨텍스트 관리 관련 선언이 포함된 헤더 파일

// 필요한 내부 구현 헤더 파일들 포함
#include "llama-impl.h"     // 내부 구현 세부사항
#include "llama-io.h"       // 입출력 관련 기능
#include "llama-mmap.h"     // 메모리 매핑 기능
#include "llama-model.h"    // 모델 구조체 및 관련 기능
#include "llama-kv-cache.h" // Key-Value 캐시 관리

// 표준 라이브러리 헤더 파일들
#include <cassert>      // 단언문(assert) 매크로
#include <cstring>      // 문자열 처리 함수
#include <stdexcept>    // 표준 예외 클래스
#include <cinttypes>    // 정수 타입 서식 지정자

//
// llama_context 클래스 구현
//

// llama_context 생성자 - 추론 컨텍스트 초기화
// model: 로드된 LLaMA 모델 인스턴스
// params: 컨텍스트 파라미터(컨텍스트 크기, 스레드 수 등)
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model) {  // 모델 참조 초기화
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    // 모델 로딩 시간 통계 복사
    t_start_us = model.t_start_us;  // 시작 시간(마이크로초)
    t_load_us  = model.t_load_us;   // 로딩에 소요된 시간(마이크로초)

    const auto & hparams = model.hparams;  // 모델 하이퍼파라미터 참조

    // 컨텍스트 파라미터 초기화 - 사용자 지정 값 또는 기본값 사용
    cparams.n_seq_max        = std::max(1u, params.n_seq_max);        // 최대 시퀀스 수(최소 1)
    cparams.n_threads        = params.n_threads;                       // 추론에 사용할 스레드 수
    cparams.n_threads_batch  = params.n_threads_batch;                 // 배치 처리에 사용할 스레드 수
    cparams.yarn_ext_factor  = params.yarn_ext_factor;                 // YaRN 확장 계수
    cparams.yarn_attn_factor = params.yarn_attn_factor;                // YaRN 어텐션 계수
    cparams.yarn_beta_fast   = params.yarn_beta_fast;                  // YaRN 빠른 보정 차원
    cparams.yarn_beta_slow   = params.yarn_beta_slow;                  // YaRN 느린 보정 차원
    cparams.defrag_thold     = params.defrag_thold;                    // KV 캐시 조각 모음 임계값
    cparams.embeddings       = params.embeddings;                      // 임베딩 계산 활성화 여부
    cparams.offload_kqv      = params.offload_kqv;                     // KQV 연산 GPU 오프로딩 여부
    cparams.flash_attn       = params.flash_attn;                      // Flash Attention 사용 여부
    cparams.no_perf          = params.no_perf;                         // 성능 측정 비활성화 여부
    cparams.pooling_type     = params.pooling_type;                    // 풀링 타입
    cparams.warmup           = false;                                  // 웜업 비활성화(초기 상태)

    // 0이나 0.0f가 전달되면 모델의 훈련 값 사용, 그렇지 않으면 사용자 지정 값 사용
    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    // YaRN 원본 컨텍스트 크기 설정 (우선순위: 파라미터 > 모델 하이퍼파라미터 > 훈련 컨텍스트 크기)
    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    // 콜백 함수 설정
    cparams.cb_eval           = params.cb_eval;                       // 평가 콜백 함수
    cparams.cb_eval_user_data = params.cb_eval_user_data;             // 콜백에 전달할 사용자 데이터

    // RoPE 스케일링 타입 설정
    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;          // 모델의 훈련 값 사용
    }

    // 스케일링 타입이 NONE이면 스케일링하지 않음 (항상 1.0)
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // 스케일링 타입이 none이면 절대 스케일링하지 않음
    }

    // YaRN 확장 계수 설정 (음수는 '설정되지 않음'을 의미)
    if (cparams.yarn_ext_factor < 0.0f) { 
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    // YaRN 어텐션 계수에 모델의 RoPE 어텐션 계수를 곱함
    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    // 풀링 타입 설정
    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;  // 기본값: 풀링 없음
        } else {
            cparams.pooling_type = hparams.pooling_type;     // 모델의 풀링 타입 사용
        }
    }

    // 인과적(causal) 어텐션 설정 - 텍스트 생성에서는 일반적으로 true
    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;  // 모델의 값 사용
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;  // 사용자 지정 값 사용
    }

    // 인과적 어텐션에서는 배치 크기가 컨텍스트 크기로 제한됨
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // 배치는 최소 GGML_KQ_MASK_PAD 크기 이상이어야 함
    // GPU 커널(예: ggml_flash_attn_ext)에서 경계 밖 접근을 방지하기 위함
    // 참조: https://github.com/ggerganov/llama.cpp/pull/5021
    if (cparams.n_batch < GGML_KQ_MASK_PAD) {
        LLAMA_LOG_WARN("%s: n_batch is less than GGML_KQ_MASK_PAD - increasing to %d\n", __func__, GGML_KQ_MASK_PAD);
        cparams.n_batch = GGML_KQ_MASK_PAD;
    }

    // 마이크로 배치 크기 설정 - 배치를 작은 단위로 나누어 처리
    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    // 시퀀스당 컨텍스트 크기 계산
    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    // 설정된 파라미터 정보 출력
    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_per_seq = %u\n",   __func__, n_ctx_per_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %d\n",   __func__, cparams.flash_attn);
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    // 시퀀스당 컨텍스트 크기가 모델의 훈련 컨텍스트 크기보다 작으면 경고
    if (n_ctx_per_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_per_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    // 시퀀스당 컨텍스트 크기가 모델의 훈련 컨텍스트 크기보다 크면 경고
    if (n_ctx_per_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_pre_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    // 모든 토큰의 로짓(확률 점수) 계산 여부
    logits_all = params.logits_all;

    // 어휘만 로드된 모델이 아닌 경우 백엔드 및 계산 자원 초기화
    if (!hparams.vocab_only) {
        // GPU 백엔드 초기화
        for (auto * dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
            }
            backends.emplace_back(backend);
        }

        // ACCEL 백엔드(예: BLAS) 추가
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // CPU 백엔드 추가
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // 백엔드에서 set_n_threads 함수 목록 생성 - 스레드 수 설정에 사용
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        // 중단 콜백 설정
        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // 그래프 출력 버퍼 초기화
        {
            // 추론 중 더 많은 출력을 사용하는 배치가 있을 때 크기 조정
            if ((uint32_t) output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // 메모리 모듈 초기화
    // 통합 KV 캐시 생성 (Key-Value 캐시는 어텐션 메커니즘의 계산 효율성을 높이는 중요한 요소)
    if (!hparams.vocab_only) {
        kv_self.reset(static_cast<llama_kv_cache_unified *>(model.create_memory()));

        LLAMA_LOG_DEBUG("%s: n_ctx = %u\n", __func__, cparams.n_ctx);

        // 컨텍스트 크기를 KV 캐시 패딩에 맞게 조정
        cparams.n_ctx = GGML_PAD(cparams.n_ctx, kv_self->get_padding(cparams));

        LLAMA_LOG_DEBUG("%s: n_ctx = %u (padded)\n", __func__, cparams.n_ctx);

        // KV 캐시 크기와 데이터 타입 설정
        uint32_t kv_size = cparams.n_ctx;
        ggml_type type_k = params.type_k;  // K(Key) 캐시 데이터 타입
        ggml_type type_v = params.type_v;  // V(Value) 캐시 데이터 타입

        // 순환 모델(Mamba 등)은 특별한 처리 필요
        if (llama_model_is_recurrent(&model)) {
            // Mamba는 언제든지 유지되는 시퀀스 수만큼의 KV 셀이 필요
            kv_size = std::max((uint32_t) 1, params.n_seq_max);
            // 상태에 대해 가능한 많은 정밀도 유지가 중요
            type_k = GGML_TYPE_F32; // Mamba의 conv_states에 ggml_ssm_conv가 요구
            type_v = GGML_TYPE_F32; // Mamba의 ssm_states에 ggml_ssm_scan이 요구
        }

        // 블록 크기 검증
        GGML_ASSERT(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
        GGML_ASSERT(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);

        // KV 캐시 초기화
        if (!kv_self->init(model, cparams, type_k, type_v, kv_size, cparams.offload_kqv)) {
            throw std::runtime_error("failed to initialize self-attention cache");
        }

        // KV 캐시 메모리 사용량 정보 출력
        {
            const size_t memory_size_k = kv_self->size_k_bytes();
            const size_t memory_size_v = kv_self->size_v_bytes();

            LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                    (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                    ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                    ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
        }
    }

    // 백엔드 초기화
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        // 백엔드 버퍼 타입 및 포인터 초기화
        backend_buft.clear();
        backend_ptrs.clear();

        // 각 백엔드의 기본 버퍼 타입 설정
        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            // CPU 백엔드이고 디바이스가 있는 경우, 중간 상태의 빠른 전송을 위해 첫 번째 디바이스의 호스트 버퍼 사용
            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                auto * dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // 최대 노드 수 계산 - 계산 그래프의 크기를 결정
        const size_t max_nodes = this->graph_max_nodes();

        LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

        // 계산 그래프와 텐서 메타데이터를 저장할 버퍼 할당
        buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

        // 파이프라인 병렬 처리 활성화 여부 결정
        // 메모리 사용량이 증가하므로 필요한 경우에만 활성화
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.params.n_gpu_layers > (int) model.hparams.n_layer &&
            model.params.split_mode == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv;

        // 파이프라인 병렬 처리는 모든 디바이스에서 비동기 계산과 이벤트 지원이 필요
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // CPU 백엔드는 무시
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // 디바이스가 비동기 계산이나 이벤트를 지원하지 않음
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        // 백엔드 스케줄러 생성 - 계산 작업을 여러 백엔드에 분배
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, pipeline_parallel));

        // 파이프라인 병렬 처리 활성화 정보 출력
        if (pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(sched.get()));
        }
    }

    // 최악의 경우 그래프 예약
    if (!hparams.vocab_only) {
        const uint32_t n_seqs = 1; // TODO: 최악의 경우 시퀀스 수
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        // llama_build_graph에서 실제로 사용되지 않지만 토큰 vs 임베딩 입력 그래프 선택에 필요
        llama_token token = model.vocab.token_bos();

        // 나중에 복원할 값 저장
        // TODO: 더 깔끔한 방식 필요
        const auto n_outputs_save = n_outputs;

        // 최대 출력 수
        n_outputs = n_tokens;

        LLAMA_LOG_DEBUG("%s: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

        int n_splits_pp = -1;  // 병렬 처리(PP) 그래프의 분할 수
        int n_nodes_pp  = -1;  // 병렬 처리 그래프의 노드 수

        int n_splits_tg = -1;  // 토큰 생성(TG) 그래프의 분할 수
        int n_nodes_tg  = -1;  // 토큰 생성 그래프의 노드 수

        // 전체 KV 캐시 시뮬레이션
        kv_self->n = kv_self->size;

        cross.v_embd.clear();

        // 버퍼가 한 번만 할당되도록 먼저 병렬 처리 그래프 예약
        {
            // 병렬 처리를 위한 마이크로 배치 설정
            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            auto * gf = graph_init();  // 그래프 초기화
            graph_build(ctx_compute.get(), gf, ubatch_pp, LLM_GRAPH_TYPE_DEFAULT);  // 그래프 구축
            if (!ggml_backend_sched_reserve(sched.get(), gf)) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }

            // 분할 수와 노드 수 저장
            n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_pp  = ggml_graph_n_nodes(gf);
        }

        // 토큰 생성 그래프로 분할 수와 노드 수 계산
        {
            // 토큰 생성을 위한 마이크로 배치 설정 (단일 토큰)
            llama_ubatch ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            auto * gf = graph_init();  // 그래프 초기화
            graph_build(ctx_compute.get(), gf, ubatch_tg, LLM_GRAPH_TYPE_DEFAULT);  // 그래프 구축
            if (!ggml_backend_sched_reserve(sched.get(), gf)) {
                throw std::runtime_error("failed to allocate compute tg buffers");
            }
            // 분할 수와 노드 수 저장
            n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_tg  = ggml_graph_n_nodes(gf);
        }

        // 추론 중 ggml-alloc 재할당을 방지하기 위해 병렬 처리 그래프로 다시 예약
        {
            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            auto * gf = graph_init();  // 그래프 초기화
            graph_build(ctx_compute.get(), gf, ubatch_pp, LLM_GRAPH_TYPE_DEFAULT);  // 그래프 구축
            if (!ggml_backend_sched_reserve(sched.get(), gf)) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        // 원래 출력 수 복원
        n_outputs = n_outputs_save;

        // 각 백엔드의 계산 버퍼 크기 정보 출력
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size > 1) {
                LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }

        // 그래프 노드 수 정보 출력
        if (n_nodes_pp == n_nodes_tg) {
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
        }

        // 그래프 분할 수 정보 출력
        if (n_splits_pp == n_splits_tg) {
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
        }
    }
}

// llama_context 소멸자 - 컨텍스트 리소스 해제
llama_context::~llama_context() = default;

// llama_context에서 모든 계산 작업을 동기화하는 함수
// 병렬 처리된 모든 작업이 완료될 때까지 대기하고 성능 통계를 업데이트함
void llama_context::synchronize() {
    // 스케줄러의 모든 작업이 완료될 때까지 대기
    ggml_backend_sched_synchronize(sched.get());

    // FIXME: 동기화 없이 여러 단일 토큰이 평가되면
    // 통계가 프롬프트 평가 통계에 추가됨
    // 이는 배치 크기 1로 배치를 평가할 때만 발생해야 함

    // 성능 통계에 평가 시간 추가
    if (n_queued_tokens == 1) {
        // 단일 토큰 평가 (생성 단계)
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us; // 평가 시간 누적(마이크로초)
        }
        n_eval++; // 단일 토큰 평가 횟수 증가
    } else if (n_queued_tokens > 1) {
        // 다중 토큰 평가 (프롬프트 처리 단계)
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us; // 프롬프트 평가 시간 누적(마이크로초)
        }
        n_p_eval += n_queued_tokens; // 프롬프트 토큰 평가 횟수 증가
    }

    // 첫 번째 평가 시 로드 시간을 더 정확하게 계산
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us; // 시작부터 첫 평가까지의 총 시간
        has_evaluated_once = true;
    }

    // 상태 초기화
    n_queued_tokens = 0;      // 대기 중인 토큰 수 초기화
    t_compute_start_us = 0;   // 계산 시작 시간 초기화
}

// 모델에 대한 참조를 반환하는 getter 함수
const llama_model & llama_context::get_model() const {
    return model;
}

// 컨텍스트 크기를 반환하는 getter 함수
// 이 값은 KV 캐시의 총 크기(최대 토큰 수)를 결정함
uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

// 시퀀스당 컨텍스트 크기를 반환 (총 컨텍스트 크기 / 최대 시퀀스 수)
// 멀티 시퀀스 처리(예: 채팅)에서 각 시퀀스에 할당된 컨텍스트 크기
uint32_t llama_context::n_ctx_per_seq() const {
    return cparams.n_ctx / cparams.n_seq_max;
}

// 배치 크기를 반환 (한 번에 처리할 수 있는 최대 토큰 수)
uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

// 마이크로 배치 크기를 반환 (내부적으로 한 번에 처리하는 토큰 수)
uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

// 최대 시퀀스 수를 반환 (동시에 처리할 수 있는 대화 흐름 수)
uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

// 추론에 사용되는 스레드 수를 반환
uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

// 배치 처리에 사용되는 스레드 수를 반환
uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

// 자체 KV 캐시 객체에 대한 포인터를 반환 (비상수 버전)
llama_kv_cache * llama_context::get_kv_self() {
    return kv_self.get();
}

// 자체 KV 캐시 객체에 대한 포인터를 반환 (상수 버전)
const llama_kv_cache * llama_context::get_kv_self() const {
    return kv_self.get();
}

// RoPE(회전 위치 임베딩)를 시프트를 적용하여 구축하는 함수
// RoPE는 트랜스포머에서 상대적 위치 정보를 인코딩하는 방법
// KV 캐시 시프트 시 위치 정보를 올바르게 조정하는 데 사용됨
ggml_tensor * llama_context::build_rope_shift(
        ggml_context * ctx0,         // GGML 컨텍스트
        ggml_tensor * cur,           // 입력 텐서 (일반적으로 키(K) 텐서)
        ggml_tensor * shift,         // 시프트 값 텐서 (각 위치의 델타 값)
        ggml_tensor * factors,       // RoPE 계수 텐서
              float   freq_base,     // 기본 주파수
              float   freq_scale,    // 주파수 스케일링 계수
        ggml_backend_buffer * bbuf) const { // 백엔드 버퍼
    // YaRN(Yet another RoPe extensioN) 관련 파라미터
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;    // 원본 컨텍스트 크기

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;  // YaRN 확장 계수
    const auto & yarn_attn_factor = cparams.yarn_attn_factor; // YaRN 어텐션 계수
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;   // YaRN 빠른 베타 계수
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;   // YaRN 느린 베타 계수

    // 모델 하이퍼파라미터
    const auto & hparams = model.hparams;

    const auto & n_rot     = hparams.n_rot;     // 회전시킬 차원 수
    const auto & rope_type = hparams.rope_type; // RoPE 유형

    ggml_tensor * tmp;

    // 양자화된 텐서인 경우 특별 처리
    if (ggml_is_quantized(cur->type)) {
        // 양자화 텐서를 f32로 변환 -> RoPE 적용 -> 다시 양자화
        tmp = ggml_cast(ctx0, cur, GGML_TYPE_F32);

        // 백엔드 버퍼가 존재하면 적절한 백엔드 설정
        if (bbuf) {
            for (const auto & backend : backends) {
                // KV 캐시가 속한 백엔드 찾기
                if (ggml_backend_supports_buft(backend.get(), ggml_backend_buffer_get_type(bbuf))) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), tmp, backend.get());
                    break;
                }
            }
        }

        // RoPE 확장 적용 (제자리 연산)
        tmp = ggml_rope_ext_inplace(ctx0, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // 결과를 원래 텐서로 복사 (양자화 형식 유지)
        tmp = ggml_cpy(ctx0, tmp, cur);
    } else {
        // 비양자화 텐서는 직접 RoPE 적용
        // 처음 n_rot 차원에만 회전 적용
        tmp = ggml_rope_ext_inplace(ctx0, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp; // 처리된 텐서 반환
}

// KV 캐시 시프트 입력을 처리하는 클래스
// 계산 그래프에 시프트 정보를 전달하는 역할
class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    // 생성자: KV 캐시에 대한 참조 저장
    llm_graph_input_k_shift(const llama_kv_cache_unified * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    // 입력 설정 메서드 (llm_graph_input_i 인터페이스 구현)
    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // [kv_size] 크기의 I32 타입 시프트 값 텐서

    // KV 캐시에 대한 참조
    const llama_kv_cache_unified * kv_self;
};

// 시프트 입력 설정 함수 구현
void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch); // 사용하지 않는 매개변수

    // k_shift 텐서가 존재하면 데이터 설정
    if (k_shift) {
        // 호스트 메모리에 있는지 확인 (CPU 메모리)
        assert(ggml_backend_buffer_is_host(k_shift->buffer));

        // 텐서 데이터 포인터
        int32_t * data = (int32_t *) k_shift->data;

        // 각 KV 셀의 델타 값을 시프트 텐서에 복사
        for (uint32_t i = 0; i < kv_self->size; ++i) {
            data[i] = kv_self->cells[i].delta;
        }
    }
}

// KV 캐시 시프트를 위한 계산 그래프를 구축하는 함수
// KV 캐시의 위치가 변경되었을 때 RoPE 위치 정보를 업데이트하는 데 사용
llm_graph_result_ptr llama_context::build_kv_self_shift(
        ggml_context * ctx0,  // GGML 컨텍스트
        ggml_cgraph * gf) const {  // 계산 그래프
    // 결과 객체 생성
    auto res = std::make_unique<llm_graph_result>();

    // 모델 하이퍼파라미터
    const auto & hparams = model.hparams;

    // 레이어 수
    const auto & n_layer = hparams.n_layer;

    // 헤드당 키 임베딩 크기
    const auto & n_embd_head_k = hparams.n_embd_head_k;
    // 헤드당 값 임베딩 크기 (사용하지 않음)
    //const auto & n_embd_head_v = hparams.n_embd_head_v;

    // KV 캐시 크기와 컨텍스트 크기가 같은지 확인 (현재 비활성화됨)
    //GGML_ASSERT(kv_self->size == n_ctx);

    // KV 시프트 입력 객체 생성
    auto inp = std::make_unique<llm_graph_input_k_shift>(kv_self.get());

    // 시프트 값을 저장할 텐서 생성 (I32 타입, 컨텍스트 크기)
    inp->k_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cparams.n_ctx);
    ggml_set_input(inp->k_shift); // 입력 텐서로 표시

    // 각 레이어에 대해 시프트 처리
    for (uint32_t il = 0; il < n_layer; ++il) {
        // KV 헤드 수 및 GQA(Grouped Query Attention)에서의 키 임베딩 크기
        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        // 슬라이딩 윈도우 어텐션(SWA) 사용 여부
        const bool is_swa = hparams.is_swa(il);

        // 레이어별 RoPE 주파수 파라미터 설정
        // SWA 사용 시 모델의 훈련 값 사용, 그렇지 않으면 컨텍스트 파라미터 사용
        const float freq_base_l  = is_swa ? hparams.rope_freq_base_train_swa  : cparams.rope_freq_base;
        const float freq_scale_l = is_swa ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;

        // RoPE 계수 텐서 가져오기
        ggml_tensor * rope_factors = kv_self->cbs.get_rope_factors(n_ctx_per_seq(), il);

        // 키(K) 텐서에 대한 뷰 생성 (3D 텐서로 해석)
        // 형태: [n_embd_head_k, n_head_kv, kv_self->size]
        ggml_tensor * k =
            ggml_view_3d(ctx0, kv_self->k_l[il],
                n_embd_head_k, n_head_kv, kv_self->size,
                ggml_row_size(kv_self->k_l[il]->type, n_embd_head_k), // 행 크기
                ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),  // 행 크기(GQA)
                0);  // 오프셋 0

        // RoPE 시프트 적용
        ggml_tensor * cur = build_rope_shift(ctx0, k, inp->k_shift, rope_factors, freq_base_l, freq_scale_l, kv_self->k_l[il]->buffer);

        // 계산 그래프에 연산 추가
        ggml_build_forward_expand(gf, cur);
    }

    // 결과 객체에 입력 추가
    res->add_input(std::move(inp));

    return res; // 구축된 그래프 결과 반환
}

// KV 캐시 조각 모음을 위한 계산 그래프를 구축하는 함수
// 캐시에서 사용되지 않는 공간을 정리하고 효율적으로 재구성
llm_graph_result_ptr llama_context::build_kv_self_defrag(
        ggml_context * ctx0,  // GGML 컨텍스트
        ggml_cgraph * gf) const {  // 계산 그래프
    // 결과 객체 생성
    auto res = std::make_unique<llm_graph_result>();

    // 모델 하이퍼파라미터
    const auto & hparams = model.hparams;

    // 조각 모음할 셀 ID 목록
    const auto & ids = kv_self->defrag_info.ids;

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(v_l[il]->type);
        const size_t v_size    = ggml_row_size (v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    // 활성화된 코드 블록 - GGML 그래프 기반 조각 모음 구현
    // 각 셀을 순회하며 필요한 이동 작업 찾기
    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];  // 대상 인덱스

        // 이동이 필요 없는 경우(이미 올바른 위치 또는 유효하지 않은 ID) 건너뜀
        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;  // 연속적으로 이동할 셀 수

        // 연속적인 셀 그룹 찾기 (최적화를 위해 일괄 처리)
        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        // 각 레이어에 대해 처리
        for (uint32_t il = 0; il < hparams.n_layer; ++il) { // NOLINT
            // 레이어별 키(K)와 값(V)의 임베딩 크기 계산
            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);  // GQA에서의 키 임베딩 크기
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);  // GQA에서의 값 임베딩 크기

            // 원본 키(K) 텐서의 일부에 대한 뷰 생성 (2D 텐서: [n_embd_k_gqa, nm])
            ggml_tensor * view_k_src = ggml_view_2d(ctx0, kv_self->k_l[il],
                    n_embd_k_gqa, nm,  // 차원 크기
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),  // 행 크기
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*i));  // 시작 오프셋

            // 대상 키(K) 텐서의 일부에 대한 뷰 생성
            ggml_tensor * view_k_dst = ggml_view_2d(ctx0, kv_self->k_l[il],
                    n_embd_k_gqa, nm,  // 차원 크기
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),  // 행 크기
                    ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa*id));  // 시작 오프셋

            ggml_tensor * view_v_src;  // 원본 값(V) 텐서 뷰
            ggml_tensor * view_v_dst;  // 대상 값(V) 텐서 뷰

            // Flash Attention 사용 여부에 따라 V 캐시 레이아웃이 다름
            if (cparams.flash_attn) {
                // Flash Attention 사용 시 V 캐시는 전치되지 않음
                view_v_src = ggml_view_2d(ctx0, kv_self->v_l[il],
                        n_embd_v_gqa, nm,  // 차원 크기 [n_embd_v_gqa, nm]
                        ggml_row_size(kv_self->v_l[il]->type, n_embd_v_gqa),  // 행 크기
                        ggml_row_size(kv_self->v_l[il]->type, n_embd_v_gqa*i));  // 시작 오프셋

                view_v_dst = ggml_view_2d(ctx0, kv_self->v_l[il],
                        n_embd_v_gqa, nm,  // 차원 크기
                        ggml_row_size(kv_self->v_l[il]->type, n_embd_v_gqa),  // 행 크기
                        ggml_row_size(kv_self->v_l[il]->type, n_embd_v_gqa*id));  // 시작 오프셋
            } else {
                // 일반 모드에서 V 캐시는 전치됨 (차원 순서 변경됨)
                view_v_src = ggml_view_2d(ctx0, kv_self->v_l[il],
                        nm, n_embd_v_gqa,  // 차원 크기 [nm, n_embd_v_gqa]
                        ggml_row_size(kv_self->v_l[il]->type, kv_self->size),  // 행 크기
                        ggml_row_size(kv_self->v_l[il]->type, i));  // 시작 오프셋

                view_v_dst = ggml_view_2d(ctx0, kv_self->v_l[il],
                        nm, n_embd_v_gqa,  // 차원 크기
                        ggml_row_size(kv_self->v_l[il]->type, kv_self->size),  // 행 크기
                        ggml_row_size(kv_self->v_l[il]->type, id));  // 시작 오프셋
            }

            // 계산 그래프에 복사 연산 추가
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));  // K 복사
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));  // V 복사
        }

        // 이미 처리된 셀 건너뛰기
        i += nm - 1;
    } 

    // 디버깅용 로그 (비활성화)
    //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);
#endif

    return res;  // 결과 객체 반환
}

// KV 캐시 업데이트 함수 - 시프트 및 조각 모음 적용
void llama_context::kv_self_update() {
    auto & kv = kv_self;  // KV 캐시 참조

    bool need_reserve = false;  // 최악의 경우 그래프 예약 필요 여부

    // 시프트가 필요한 경우
    if (kv->has_shift) {
        // 시프트 기능 지원 확인
        if (!kv->get_can_shift()) {
            GGML_ABORT("The current context does not support K-shift");  // 지원하지 않으면 중단
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);  // 디버그 로그

        // RoPE 유형이 NONE이 아닌 경우에만 K-시프트 적용
        if (model.hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched.get());  // 스케줄러 초기화

            auto * gf = graph_init();  // 새 계산 그래프 초기화

            // KV 시프트를 위한 그래프 구축
            auto res = build_kv_self_shift(ctx_compute.get(), gf);

            // 그래프를 위한 메모리 할당
            ggml_backend_sched_alloc_graph(sched.get(), gf);

            // 입력 설정
            res->set_inputs(nullptr);

            // 그래프 계산 실행 (동기화 없이)
            graph_compute(gf, false);

            need_reserve = true;  // 리소스 예약 필요 표시
        }

        // 시프트 상태 초기화
        {
            kv->has_shift = false;  // 시프트 완료 표시

            // 모든 셀의 델타 값 초기화
            for (uint32_t i = 0; i < kv->size; ++i) {
                kv->cells[i].delta = 0;
            }
        }
    }

    // KV 캐시 조각 모음이 필요한 경우
    if (kv->do_defrag) {
        LLAMA_LOG_DEBUG("%s: defragmenting KV cache\n", __func__);  // 디버그 로그

        // 전체 defrag 과정 시간 측정 시작
        const int64_t t_defrag_start = ggml_time_us();

        // 조각 모음 준비 (그래프 노드 수 제한 확인)
        if (kv->defrag_prepare(graph_max_nodes())) {
            // 그래프 빌드 및 계산 시간 측정 시작
            const int64_t t_graph_start = ggml_time_us();

            ggml_backend_sched_reset(sched.get());  // 스케줄러 초기화

            auto * gf = graph_init();  // 새 계산 그래프 초기화

            // 조각 모음을 위한 그래프 구축
            auto res = build_kv_self_defrag(ctx_compute.get(), gf);

            // 그래프를 위한 메모리 할당
            ggml_backend_sched_alloc_graph(sched.get(), gf);

            // 입력 설정
            res->set_inputs(nullptr);

            // 그래프 계산 실행 (동기화 없이)
            graph_compute(gf, false);

            // 그래프 빌드 및 계산 시간 측정 종료
            const int64_t t_graph_end = ggml_time_us();
            LLAMA_LOG_INFO("%s: defrag graph build and compute took %.3f ms\n", __func__, (t_graph_end - t_graph_start) / 1000.0f);

            need_reserve = true;  // 리소스 예약 필요 표시
        }

        kv->do_defrag = false;  // 조각 모음 완료 표시

        // 전체 defrag 과정 시간 측정 종료
        const int64_t t_defrag_end = ggml_time_us();
        LLAMA_LOG_INFO("%s: total defragmentation took %.3f ms\n", __func__, (t_defrag_end - t_defrag_start) / 1000.0f);
    }

    // 필요한 경우 최악의 경우 그래프 예약
    if (need_reserve) {
        LLAMA_LOG_DEBUG("%s: reserving a worst case graph\n", __func__);  // 디버그 로그

        // 최악의 경우 그래프 구축 매개변수
        uint32_t n_seqs = 1;  // TODO: 최악의 경우 시퀀스 수
        uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);  // 토큰 수 (컨텍스트 크기와 마이크로 배치 크기 중 작은 값)

        // 전체 KV 캐시 시뮬레이션
        kv_self->n = kv_self->size;  // KV 캐시가 가득 찬 것으로 가정

        // 입력 토큰 (실제로 사용되지 않지만 토큰과 임베딩 입력 그래프 중 선택하는 데 필요)
        llama_token token = model.vocab.token_bos();
        // 마이크로 배치 구성
        llama_ubatch ubatch = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

        // 그래프 초기화 및 구축
        auto * gf = graph_init();
        graph_build(ctx_compute.get(), gf, ubatch, LLM_GRAPH_TYPE_DEFAULT);

        // 최악의 경우 그래프로 스케줄러 초기화
        ggml_backend_sched_reset(sched.get());
        if (!ggml_backend_sched_reserve(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);  // 버퍼 할당 실패 로그
        }
    }
}

// 풀링 타입을 반환하는 getter 함수
// 임베딩 결과를 풀링하는 방식 결정 (none, mean, cls 등)
enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

// 로짓(logits) 배열에 대한 포인터 반환 (모든 토큰에 대한 확률 점수)
float * llama_context::get_logits() {
    // 로짓 재정렬 (역방향 호환성을 위해)
    output_reorder();

    return logits;  // 로짓 배열 포인터 반환
}

// 특정 인덱스의 로짓 배열에 대한 포인터 반환
float * llama_context::get_logits_ith(int32_t i) {
    int32_t j = -1;  // 실제 출력 배열 인덱스

    try {
        // 로짓 배열이 없으면 예외 발생
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        // 음수 인덱스 처리 (파이썬 스타일 뒤에서부터 접근)
        if (i < 0) {
            j = n_outputs + i;  // 뒤에서부터 접근
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } 
        // 범위 검사
        else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } 
        // 양수 인덱스는 출력 ID 배열에서 실제 인덱스 조회
        else {
            j = output_ids[i];
        }

        // 유효한 로짓 인덱스인지 확인
        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        // 출력 범위 내인지 확인
        if (j >= n_outputs) {
            // 이 경우는 발생하지 않아야 함 (내부 오류)
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, n_outputs));
        }

        // 해당 인덱스의 로짓 포인터 반환 (각 토큰의 확률 점수)
        return logits + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        // 오류 발생 시 로그 출력
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        // 디버그 모드에서는 즉시 중단
        GGML_ABORT("fatal error");
#else
        // 릴리스 모드에서는 NULL 반환
        return nullptr;
#endif
    }
}

// 임베딩 배열에 대한 포인터 반환 (토큰의 벡터 표현)
float * llama_context::get_embeddings() {
    // 임베딩 재정렬 (역방향 호환성을 위해)
    output_reorder();

    return embd;  // 임베딩 배열 포인터 반환
}

// 특정 인덱스의 임베딩 배열에 대한 포인터 반환
float * llama_context::get_embeddings_ith(int32_t i) {
    int32_t j = -1;  // 실제 출력 배열 인덱스

    try {
        // 임베딩 배열이 없으면 예외 발생
        if (embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        // 음수 인덱스 처리 (파이썬 스타일 뒤에서부터 접근)
        if (i < 0) {
            j = n_outputs + i;  // 뒤에서부터 접근
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } 
        // 범위 검사
        else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } 
        // 양수 인덱스는 출력 ID 배열에서 실제 인덱스 조회
        else {
            j = output_ids[i];
        }

        // 유효한 임베딩 인덱스인지 확인
        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        // 출력 범위 내인지 확인
        if (j >= n_outputs) {
            // 이 경우는 발생하지 않아야 함 (내부 오류)
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, n_outputs));
        }

        // 해당 인덱스의 임베딩 포인터 반환 (각 토큰의 벡터 표현)
        return embd + j*model.hparams.n_embd;
    } catch (const std::exception & err) {
        // 오류 발생 시 로그 출력
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        // 디버그 모드에서는 즉시 중단
        GGML_ABORT("fatal error");
#else
        // 릴리스 모드에서는 NULL 반환
        return nullptr;
#endif
    }
}

// 특정 시퀀스 ID에 대한 임베딩 벡터를 반환하는 함수
// 시퀀스 ID를 기반으로 해당 시퀀스의 임베딩을 조회
float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    // embd_seq 맵에서 주어진 시퀀스 ID를 찾음
    auto it = embd_seq.find(seq_id);
    
    // 시퀀스 ID가 존재하지 않으면 nullptr 반환
    if (it == embd_seq.end()) {
        return nullptr;
    }

    // 해당 시퀀스의 임베딩 벡터 데이터 포인터 반환
    return it->second.data();
}

// 스레드풀을 컨텍스트에 연결하는 함수
// 병렬 계산을 위한 외부 스레드풀 설정
void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,        // 주 연산용 스레드풀
           ggml_threadpool_t threadpool_batch) { // 배치 처리용 스레드풀 (선택적)
    LLAMA_LOG_DEBUG("%s: call\n", __func__);    // 디버그 로그 출력

    // 주 연산용 스레드풀 설정
    this->threadpool = threadpool;
    
    // 배치 처리용 스레드풀 설정 (제공되지 않았으면 주 스레드풀 사용)
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

// 스레드풀 연결 해제 함수
// 컨텍스트에서 스레드풀 참조를 제거
void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);    // 디버그 로그 출력

    // 스레드풀 참조 제거
    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

// 스레드 수 설정 함수
// 계산에 사용할 스레드 수를 지정
void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);    // 디버그 로그 출력

    // 컨텍스트 매개변수에 스레드 수 저장
    cparams.n_threads       = n_threads;       // 주 연산용 스레드 수
    cparams.n_threads_batch = n_threads_batch; // 배치 처리용 스레드 수
}

// 중단 콜백 설정 함수
// 계산 중 중단 가능성을 제공하는 콜백 함수 설정
void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);    // 디버그 로그 출력

    // 중단 콜백 함수와 관련 데이터 저장
    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    // 모든 백엔드에 중단 콜백 전파
    for (auto & backend : backends) {
        // 백엔드 레지스트리 얻기
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        
        // 중단 콜백 설정 함수 포인터 얻기
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        
        // 함수가 존재하면 백엔드에 콜백 설정
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
        }
    }
}

// 임베딩 모드 설정 함수
// 토큰 임베딩 계산 여부 설정
void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);    // 디버그 로그 출력

    // 임베딩 계산 여부 설정
    cparams.embeddings = value;
}

// 인과적 어텐션(causal attention) 설정 함수
// 인과적 어텐션 사용 여부 설정 (미래 토큰을 보지 않는 마스킹)
void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);    // 디버그 로그 출력

    // 인과적 어텐션 사용 여부 설정
    cparams.causal_attn = value;
}

// 웜업 모드 설정 함수
// 모델 웜업 여부 설정 (첫 실행 시 성능 최적화)
void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);    // 디버그 로그 출력

    // 웜업 모드 설정
    cparams.warmup = value;
}

// LoRA 어댑터 설정 함수
// 저차원 랭크 적응(Low-Rank Adaptation) 어댑터 추가 및 스케일 설정
void llama_context::set_adapter_lora(
            llama_adapter_lora * adapter,  // LoRA 어댑터 포인터
            float scale) {                // 적용할 스케일(가중치)
    LLAMA_LOG_DEBUG("%s: adapter = %p, scale = %f\n", __func__, (void *) adapter, scale);    // 디버그 로그 출력

    // LoRA 어댑터와 스케일을 맵에 저장
    loras[adapter] = scale;
}

// LoRA 어댑터 제거 함수
// 특정 LoRA 어댑터를 컨텍스트에서 제거
bool llama_context::rm_adapter_lora(
            llama_adapter_lora * adapter) {  // 제거할 LoRA 어댑터 포인터
    LLAMA_LOG_DEBUG("%s: adapter = %p\n", __func__, (void *) adapter);    // 디버그 로그 출력

    // 어댑터 맵에서 검색
    auto pos = loras.find(adapter);
    
    // 어댑터가 존재하면 제거하고 true 반환
    if (pos != loras.end()) {
        loras.erase(pos);
        return true;
    }

    // 어댑터가 없으면 false 반환
    return false;
}

// 모든 LoRA 어댑터 제거 함수
// 컨텍스트에 등록된 모든 LoRA 어댑터 초기화
void llama_context::clear_adapter_lora() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);    // 디버그 로그 출력

    // LoRA 어댑터 맵 초기화
    loras.clear();
}

// 어댑터 컨트롤 벡터 적용 함수
// 특정 레이어 범위에 컨트롤 벡터 적용 (모델 제어)
bool llama_context::apply_adapter_cvec(
            const float * data,    // 컨트롤 벡터 데이터
                 size_t   len,     // 데이터 길이
                int32_t   n_embd,  // 임베딩 차원 수
                int32_t   il_start,// 시작 레이어 인덱스
                int32_t   il_end) {// 종료 레이어 인덱스
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);    // 디버그 로그 출력

    // 컨트롤 벡터 적용 후 성공 여부 반환
    return cvec.apply(model, data, len, n_embd, il_start, il_end);
}

// 토큰 인코딩 함수
// 입력 토큰 배치를 인코딩하여 임베딩 생성
int llama_context::encode(llama_batch & inp_batch) {
    // 입력 토큰이 없으면 오류 반환
    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // 필요한 경우 입력 배치용 임시 메모리 할당
    // TODO: 다중 시퀀스의 경우 pos_max()가 모든 시퀀스 중 최대값이므로 부정확할 수 있음
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : kv_self->pos_max() + 1);

    // 할당된 배치 참조
    const llama_batch & batch = batch_allocr.batch;
    const int32_t n_tokens = batch.n_tokens;  // 총 토큰 수

    // 모델 하이퍼파라미터 참조
    const auto & hparams = model.hparams;

    // 토큰과 임베딩이 둘 다 있거나 둘 다 없으면 안 됨 (둘 중 하나만 제공해야 함)
    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    // 토큰 ID가 제공된 경우 유효한 범위인지 검사
    if (batch.token) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return -1;
            }
        }
    }

    // 비인과적 인코딩은 마이크로 배칭이 불가능하므로 배치를 한 번에 처리
    // 인코더는 n_ubatch가 n_tokens 이상이어야 함
    GGML_ASSERT(cparams.n_ubatch >= (uint32_t) n_tokens && "encoder requires n_ubatch >= n_tokens");

    // 계산 시작 시간 기록 (아직 기록되지 않은 경우)
    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // 대기 중인 토큰 수 증가
    n_queued_tokens += n_tokens;

    // 임베딩 차원 수
    const int64_t n_embd = hparams.n_embd;

    // 배치를 단순 분할 형태의 sbatch로 변환 (모든 토큰에 대해 로짓 계산)
    sbatch.from_batch(batch, n_embd, /* simple_split */ true, /* logits_all */ true);

    // 단순 분할 마이크로 배치 생성
    const llama_ubatch ubatch = sbatch.split_simple(n_tokens);

    // 출력 버퍼 예약
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;  // 메모리 할당 실패
    };

    // 출력 ID 매핑 설정 (순차적으로)
    for (int32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    // 출력 수 설정
    n_outputs = n_tokens;

    // 배치 매니저 준비 (주석 처리됨)
    //batch_manager->prepare(ubatch);

    // 스케줄러 초기화 및 평가 콜백 설정
    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

    // 계산 그래프 초기화
    auto * gf = graph_init();
    
    // 인코더 타입의 계산 그래프 구축
    auto res = graph_build(ctx_compute.get(), gf, ubatch, LLM_GRAPH_TYPE_ENCODER);

    // 그래프에 메모리 할당
    ggml_backend_sched_alloc_graph(sched.get(), gf);

    // 입력 설정
    res->set_inputs(&ubatch);

    // 그래프 계산 실행 (여러 토큰이면 병렬 처리)
    const auto compute_status = graph_compute(gf, n_tokens > 1);
    
    // 계산 상태에 따른 처리
    switch (compute_status) {
        case GGML_STATUS_SUCCESS:
            break;  // 성공적으로 계산 완료
        case GGML_STATUS_ABORTED:
            return 2;  // 사용자에 의해 중단됨
        case GGML_STATUS_ALLOC_FAILED:
            return -2;  // 메모리 할당 실패
        case GGML_STATUS_FAILED:
        default:
            return -3;  // 기타 실패
    }

    // 임베딩 텐서 (풀링된 임베딩이 있으면 사용, 없으면 일반 임베딩 사용)
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

    // 임베딩 추출
    if (t_embd) {
        // 임베딩 텐서의 백엔드 가져오기
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        GGML_ASSERT(embd != nullptr);

        // 풀링 타입에 따라 다른 방식으로 임베딩 처리
        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // 토큰별 임베딩 추출
                    GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd, 0, n_tokens*n_embd*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:  // 평균 풀링
            case LLAMA_POOLING_TYPE_CLS:   // CLS 토큰 풀링
            case LLAMA_POOLING_TYPE_LAST:  // 마지막 토큰 풀링
                {
                    // 시퀀스별 임베딩 추출
                    auto & embd_seq_out = embd_seq;
                    embd_seq_out.clear();

                    // 현재 동일 시퀀스 처리는 미구현 상태
                    GGML_ASSERT(!ubatch.equal_seqs); // TODO: handle equal splits

                    // 각 토큰에 대해 시퀀스별 임베딩 추출
                    for (int32_t i = 0; i < n_tokens; i++) {
                        const llama_seq_id seq_id = ubatch.seq_id[i][0];
                        
                        // 이미 처리된 시퀀스는 건너뜀
                        if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                            continue;
                        }
                        
                        // 새 시퀀스 임베딩 공간 할당 및 데이터 복사
                        embd_seq_out[seq_id].resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:  // 랭크 기반 풀링
                {
                    // TODO: 아직 구현되지 않음, PR #9510 참조
                    // https://github.com/ggerganov/llama.cpp/pull/9510
                    GGML_ABORT("RANK pooling not implemented yet");
                }
            case LLAMA_POOLING_TYPE_UNSPECIFIED:  // 미지정 풀링 타입
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // 백엔드 동기화 전에 다음 토큰을 위해 상태 초기화
    // 이렇게 하면 초기화 CPU 작업이 디바이스 계산과 겹칠 수 있음
    ggml_backend_sched_reset(sched.get());

    // T5 아키텍처를 위한 특별 처리 (임시 해결책)
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;  // 주석 처리됨

        // 교차 어텐션용 임베딩 차원 및 인코더 길이 저장
        cross.n_embd = t_embd->ne[0];  // 임베딩 차원
        cross.n_enc  = t_embd->ne[1];  // 인코더 길이
        
        // 교차 어텐션용 임베딩 버퍼 할당 및 데이터 복사
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd, ggml_nbytes(t_embd));

        // 인코딩 중 사용된 시퀀스 ID 기억 (나중에 교차 어텐션에 필요)
        cross.seq_ids_enc.resize(n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) {
            for (int s = 0; s < ubatch.n_seq_id[i]; s++) {
                llama_seq_id seq_id = ubatch.seq_id[i][s];
                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    // 성공적으로 완료됨을 나타내는 0 반환
    return 0;
}

// 토큰 디코딩 함수 - 주어진 입력 배치를 처리하여 로짓(다음 토큰 확률)이나 임베딩 생성
// LLM 모델의 핵심 추론 함수
int llama_context::decode(llama_batch & inp_batch) {
    // 빈 배치 검사 - 토큰이 없으면 오류 반환
    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // 임시 메모리 할당 (필요한 경우)
    // pos가 제공되지 않았으면 현재 KV 캐시의 최대 위치 + 1부터 시작
    // TODO: 다중 시퀀스에서는 pos_max()가 모든 시퀀스의 최대값이므로 부정확할 수 있음
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : kv_self->pos_max() + 1);

    // 할당된 배치 참조 저장
    const llama_batch & batch = batch_allocr.batch;

    // 모델의 어휘 사전과 하이퍼파라미터 참조
    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    // 어휘 사전 크기 (가능한 토큰 수)
    const int32_t n_vocab = vocab.n_tokens();

    // 총 토큰 수와 임베딩 차원 수
    const int64_t n_tokens_all = batch.n_tokens;
    const int64_t n_embd       = hparams.n_embd;

    // 배치 가드 클래스 - KV 캐시 슬롯 복원을 위한 RAII 패턴 구현
    // TODO: 나중에 제거 예정
    class batch_guard {
    public:
        // 생성자: KV 캐시 참조 저장
        batch_guard(llama_kv_cache_unified & kv_self) : kv_slot_restorer(kv_self) {
        }

        // 소멸자: 작업이 완료되지 않았으면 KV 캐시 슬롯 복원
        ~batch_guard() {
            if (!is_done) {
                kv_slot_restorer.restore();
            }
        }

        // 작업 완료 표시 (소멸자에서 복원 방지)
        void done() {
            is_done = true;
        }

        // KV 캐시 슬롯 정보 저장
        void save(const llama_kv_cache_slot_info & slot_info) {
            kv_slot_restorer.save(slot_info);
        }

    private:
        bool is_done = false;  // 작업 완료 여부

        llama_kv_slot_restorer kv_slot_restorer;  // KV 슬롯 복원기
    };

    // 배치 가드 인스턴스 생성
    batch_guard bg(*kv_self);

    // 배치에는 토큰 또는 임베딩 중 하나만 제공되어야 함 (둘 다 제공 불가)
    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    // 토큰 ID 유효성 검사 (유효 범위: 0 ~ 어휘 크기-1)
    if (batch.token) {
        for (int64_t i = 0; i < n_tokens_all; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%" PRId64 "] = %d\n", __func__, i, batch.token[i]);
                throw std::runtime_error("invalid token");
            }
        }
    }

    // 총 토큰 수가 배치 크기 제한을 초과하지 않는지 확인
    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    // 비인과적 어텐션은 마이크로 배치 크기가 토큰 수 이상이어야 함
    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // 계산 시작 시간 기록 (첫 호출 시)
    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    // 대기 중인 토큰 수 증가
    n_queued_tokens += n_tokens_all;

    // 풀링된 임베딩 모드 여부 - 임베딩 모드이면서 NONE이 아닌 풀링 타입 사용 시
    // 이 경우 배치.로짓은 무시하고 모든 토큰 출력
    const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

    // 시퀀스별 임베딩 맵 초기화
    embd_seq.clear();

    // 총 출력 수 초기화
    int64_t n_outputs_all = 0;

    // 출력 수 계산
    if (batch.logits && !embd_pooled) {
        // logits 플래그가 설정된 토큰만 출력으로 카운트
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            n_outputs_all += batch.logits[i] != 0;
        }
    } else if (logits_all || embd_pooled) {
        // 모든 토큰 출력 (logits_all이 true이거나 풀링된 임베딩 모드)
        n_outputs_all = n_tokens_all;
    } else {
        // 마지막 출력만 유지 (기본 모드)
        n_outputs_all = 1;
    }

    // 모든 토큰에 대해 로짓을 계산하는지 여부
    const bool logits_all = n_outputs_all == n_tokens_all;

    // 배치를 sbatch(split batch) 형식으로 변환
    // - simple_split: 비순환 모델은 단순 분할 사용
    // - logits_all: 모든 토큰에 대해 로짓 계산 여부
    sbatch.from_batch(batch, n_embd,
            /* simple_split */ !kv_self->recurrent,
            /* logits_all   */ logits_all);

    // 출력 버퍼 예약 - 필요한 메모리 공간 할당
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %" PRId64 " outputs\n", __func__, n_outputs_all);
        return -2;  // 메모리 할당 실패
    };

    // 이전 단계에서 처리된 출력 수 추적
    int64_t n_outputs_prev = 0;

    // sbatch에 남은 토큰이 있으면 계속 처리
    while (sbatch.n_tokens > 0) {
        // 마이크로 배치 초기화
        llama_ubatch ubatch = llama_ubatch();

        // 마이크로 배치 크기 참조
        const auto & n_ubatch = cparams.n_ubatch;

        // 모델 아키텍처에 따라 다른 분할 방식 사용
        if (kv_self->recurrent) {
            if (embd_pooled) {
                // 풀링된 임베딩은 마이크로 배치 간에 분할할 수 없음 (아직 미구현)
                ubatch = sbatch.split_seq(cparams.n_ubatch);
            } else {
                // 순환 모델 아키텍처는 동일 길이의 시퀀스로 구현하기 쉬움
                ubatch = sbatch.split_equal(cparams.n_ubatch);
            }
        } else {
            // 비순환 모델은 단순 분할 사용
            ubatch = sbatch.split_simple(n_ubatch);
        }

        // 현재 마이크로 배치의 출력 수 계산
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                // 모든 토큰이 출력 대상이면 토큰 수 = 출력 수
                n_outputs_new = ubatch.n_tokens;
            } else {
                // 특정 토큰만 출력 대상이면 출력 플래그 확인
                GGML_ASSERT(ubatch.output);
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // 출력 수 설정 (그래프 구축 전에 설정 필요)
            n_outputs = n_outputs_new;
        }

        // 비인과적 마스크는 KV 캐시를 사용하지 않음
        if (hparams.causal_attn) {
            // KV 캐시 업데이트 - 시프트 및 조각 모음 적용
            kv_self_update();

            // 현재 헤드 위치 이전에 충분한 미사용 셀이 있으면
            // 캐시 시작부터 채우기 위해 헤드 위치 재설정
            if (kv_self->head > kv_self->used + 2*ubatch.n_tokens) {
                kv_self->head = 0;
            }

            // KV 캐시에서 마이크로 배치를 위한 슬롯 찾기
            const auto slot_info = kv_self->find_slot(ubatch);
            
            if (!slot_info) {
                LLAMA_LOG_ERROR("%s: failed to prepare ubatch\n", __func__);
                return -3;  // 슬롯 할당 실패
            }

            // 슬롯 정보 저장 (나중에 복원 가능하도록)
            bg.save(slot_info);

            if (!kv_self->recurrent) {
                // 휴리스틱: 캐시가 아직 완전히 활용되지 않았으면 전체 캐시에 어텐션하지 않음
                // 충분한 생성 후에는 이 휴리스틱의 이점이 사라짐
                // 캐시 조각 모음을 시작하면 이 기능의 중요성이 더 커짐
                const uint32_t pad = kv_self->get_padding(cparams);
                kv_self->n = std::min(kv_self->size, std::max(pad, GGML_PAD(kv_self->cell_max(), pad)));
            }
        }

        // KV 캐시 상태 디버깅용 (주석 처리됨)
        //printf("kv_self.n = %5d, kv_self.used = %5d, kv_self.head = %5d\n", kv_self->n, kv_self->used, kv_self->head);

        // 스케줄러 초기화 및 평가 콜백 설정
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        // 계산 그래프 초기화
        auto * gf = graph_init();
        // 디코더 타입의 계산 그래프 구축
        auto res = graph_build(ctx_compute.get(), gf, ubatch, LLM_GRAPH_TYPE_DECODER);

        // 그래프 구축 시간 디버깅용 (주석 처리됨)
        // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

        // 그래프에 메모리 할당
        ggml_backend_sched_alloc_graph(sched.get(), gf);

        // 입력 설정
        res->set_inputs(&ubatch);

        // 그래프 계산 실행 (여러 토큰이면 병렬 처리)
        const auto compute_status = graph_compute(gf, ubatch.n_tokens > 1);
        if (compute_status != GGML_STATUS_SUCCESS) {
            // 계산 상태에 따른 오류 반환
            switch (compute_status) {
                case GGML_STATUS_ABORTED:
                    return 2;  // 사용자에 의해 중단됨
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;  // 메모리 할당 실패
                case GGML_STATUS_FAILED:
                default:
                    return -3;  // 기타 실패
            }
        }
        //내가 추가 ----------------------------------------
        // LLAMA_LOG_INFO("\nubatch n_tokens %d\n", ubatch.n_tokens);
        // LLAMA_LOG_INFO("ubatch n_seq_tokens %d\n", ubatch.n_seq_tokens);
        // LLAMA_LOG_INFO("ubatch n_seqs %d\n\n", ubatch.n_seqs);

        // KV 링 버퍼 헤드 위치 업데이트
        {
            // 처리된 토큰 수만큼 헤드 위치 증가
            kv_self->head += ubatch.n_tokens;

            // KV 캐시 헤드가 유효한 인덱스를 가리키도록 보장
            // 크기를 초과하면 처음으로 돌아감 (순환 버퍼)
            if (kv_self->head >= kv_self->size) {
                kv_self->head = 0;
            }
        }

        // 계산 그래프를 DOT 형식으로 덤프 (디버깅용, 주석 처리됨)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        // 로짓 텐서와 임베딩 텐서 참조 얻기
        auto * t_logits = cparams.embeddings ? nullptr         : res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

        // 풀링된 임베딩이 있으면 해당 텐서 사용
        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // 로짓 추출 (토큰 확률 분포)
        if (t_logits && n_outputs > 0) {
            // 로짓 텐서의 백엔드 얻기
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits != nullptr);

            // 출력 로짓 버퍼 포인터 계산
            float * logits_out = logits + n_outputs_prev*n_vocab;

            // 출력이 있으면 로짓 복사
            if (n_outputs) {
                // 출력 인덱스 유효성 확인
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                // 메모리 크기 유효성 확인
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits_size);
                // 로짓 데이터 비동기 복사
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // 임베딩 추출 (토큰 벡터 표현)
        if (t_embd && n_outputs > 0) {
            // 임베딩 텐서의 백엔드 얻기
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            // 풀링 타입에 따라 다른 방식으로 임베딩 처리
            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // 토큰별 임베딩 추출
                        GGML_ASSERT(embd != nullptr);
                        // 출력 임베딩 버퍼 포인터 계산
                        float * embd_out = embd + n_outputs_prev*n_embd;

                        // 출력이 있으면 임베딩 복사
                        if (n_outputs) {
                            // 출력 인덱스 유효성 확인
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            // 메모리 크기 유효성 확인
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size);
                            // 임베딩 데이터 비동기 복사
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:  // 평균 풀링
                case LLAMA_POOLING_TYPE_CLS:   // CLS 토큰 풀링
                case LLAMA_POOLING_TYPE_LAST:  // 마지막 토큰 풀링
                    {
                        // 시퀀스별 임베딩 추출 (배치마다 처리 전 초기화됨)
                        auto & embd_seq_out = embd_seq;

                        // 각 시퀀스에 대해 임베딩 추출
                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            // 시퀀스 ID 가져오기
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            // 이미 처리된 시퀀스는 건너뜀
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            // 새 시퀀스 임베딩 공간 할당
                            embd_seq_out[seq_id].resize(n_embd);
                            // 임베딩 데이터 비동기 복사
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:  // 랭크 기반 풀링 (재순위화 점수)
                    {
                        // 시퀀스당 하나의 재순위화 점수 추출 (단일 부동소수점)
                        auto & embd_seq_out = embd_seq;

                        // 각 시퀀스에 대해 점수 추출
                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            // 시퀀스 ID 가져오기
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            // 이미 처리된 시퀀스는 건너뜀
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            // 재순위화 점수용 공간 할당 (1개 부동소수점)
                            embd_seq_out[seq_id].resize(1);
                            // 점수 데이터 비동기 복사
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (seq_id)*sizeof(float), sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:  // 미지정 풀링 타입
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // 이전 단계에서 처리된 출력 수 업데이트
        n_outputs_prev += n_outputs;
    }

    // 배치 처리 완료 표시
    bg.done();

    // 출력 매핑 설정 (출력 인덱스 관리)
    {
        // 출력이 정렬되어 있는지 확인
        bool sorted_output = true;

        // 출력 ID 수가 총 출력 수와 일치하는지 확인
        GGML_ASSERT(sbatch.out_ids.size() == (size_t) n_outputs_all);

        // 각 출력에 대해 매핑 설정
        for (int64_t i = 0; i < n_outputs_all; ++i) {
            // 출력 ID 가져오기
            int64_t out_id = sbatch.out_ids[i];
            // 출력 ID 매핑 설정
            output_ids[out_id] = i;
            // 순서가 변경되었는지 확인
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // 출력이 정렬되어 있으면 ID 배열 정리 (메모리 절약)
        if (sorted_output) {
            sbatch.out_ids.clear();
        }
    }

    // 총 출력 수 설정 (llama_get_logits_ith 함수에서 사용)
    n_outputs = n_outputs_all;

    // 계산 완료 대기 (모델 출력 가져올 때 자동으로 수행)
    //synchronize();

    // KV 캐시 조각 모음 필요성 결정
    if (cparams.causal_attn && cparams.defrag_thold > 0.0f) {
        // LLAMA_LOG_INFO("defrag_thold : %f\n", cparams.defrag_thold);
        // - 작은 컨텍스트는 조각 모음하지 않음 (2048 토큰 미만)
        // - 패딩도 사용된 토큰 수에 포함
        
        // 조각화 감지 시작 시간 측정
        const auto t_frag_check_start = ggml_time_us();
        
        // 조각화 비율 계산: 미사용 셀 비율
        const float fragmentation = kv_self->n < 2048 ? 
            std::max(0.0f, 1.0f - float(kv_self->used + kv_self->get_padding(cparams))/float(kv_self->n)) : 0.0f;

        // 조각화 비율이 임계값을 초과하면 조각 모음 요청
        if (fragmentation > cparams.defrag_thold) {
            // 조각화 감지 종료 시간 측정
            const auto t_frag_check_end = ggml_time_us();
            LLAMA_LOG_INFO("%s: fragmentation check took %.3f ms\n", __func__, (t_frag_check_end - t_frag_check_start) / 1000.0f);
            
            LLAMA_LOG_INFO("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);

            // 다음 llama_kv_cache_update 호출 시 조각 모음 수행하도록 표시
            kv_self->defrag();
        }
    }

    // 다음 토큰을 위한 상태 초기화 (백엔드 동기화 전)
    // CPU 활동이 디바이스 계산과 겹치도록 함
    ggml_backend_sched_reset(sched.get());

    // 성공적으로 완료됨을 나타내는 0 반환
    return 0;
}

//
// output
//

int32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = vocab.n_tokens();
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    bool has_logits = !cparams.embeddings;
    bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            buf_output = nullptr;
            logits = nullptr;
            embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    logits = has_logits ? output_base               : nullptr;
    embd   = has_embd   ? output_base + logits_size : nullptr;

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    ggml_backend_buffer_clear(buf_output.get(), 0);

    this->n_outputs     = 0;
    this->n_outputs_max = n_outputs_max;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    auto & out_ids = sbatch.out_ids;
    if (!out_ids.empty()) {
        const uint32_t n_vocab = model.vocab.n_tokens();
        const uint32_t n_embd  = model.hparams.n_embd;

        GGML_ASSERT((size_t) n_outputs == out_ids.size());

        // TODO: is there something more efficient which also minimizes swaps?
        // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
        for (int32_t i = 0; i < n_outputs - 1; ++i) {
            int32_t j_min = i;
            for (int32_t j = i + 1; j < n_outputs; ++j) {
                if (out_ids[j] < out_ids[j_min]) {
                    j_min = j;
                }
            }
            if (j_min == i) { continue; }
            std::swap(out_ids[i], out_ids[j_min]);
            if (logits_size > 0) {
                for (uint32_t k = 0; k < n_vocab; k++) {
                    std::swap(logits[i*n_vocab + k], logits[j_min*n_vocab + k]);
                }
            }
            if (embd_size > 0) {
                for (uint32_t k = 0; k < n_embd; k++) {
                    std::swap(embd[i*n_embd + k], embd[j_min*n_embd + k]);
                }
            }
        }
        std::fill(output_ids.begin(), output_ids.end(), -1);
        for (int32_t i = 0; i < n_outputs; ++i) {
            output_ids[out_ids[i]] = i;
        }
        out_ids.clear();
    }
}

//
// graph
//

int32_t llama_context::graph_max_nodes() const {
    return std::max<int32_t>(65536, 5*model.n_tensors());
}

ggml_cgraph * llama_context::graph_init() {
    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    return ggml_new_graph_custom(ctx_compute.get(), graph_max_nodes(), false);
}

llm_graph_result_ptr llama_context::graph_build(
            ggml_context * ctx,
             ggml_cgraph * gf,
      const llama_ubatch & ubatch,
            llm_graph_type gtype) {
    return model.build_graph(
            {
                /*.ctx         =*/ ctx,
                /*.arch        =*/ model.arch,
                /*.hparams     =*/ model.hparams,
                /*.cparams     =*/ cparams,
                /*.ubatch      =*/ ubatch,
                /*.sched       =*/ sched.get(),
                /*.backend_cpu =*/ backend_cpu,
                /*.cvec        =*/ &cvec,
                /*.loras       =*/ &loras,
                /*.memory      =*/ kv_self.get(),
                /*.cross       =*/ &cross,
                /*.n_outputs   =*/ n_outputs,
                /*.cb          =*/ graph_get_cb(),
            }, gf, gtype);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        set_threadpool_fn(backend_cpu, tp);
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.params.n_gpu_layers > (int) model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    // write output ids
    {
        LLAMA_LOG_DEBUG("%s: - writing output ids\n", __func__);

        output_reorder();

        const auto n_outputs    = this->n_outputs;
        const auto & output_ids = this->output_ids;

        std::vector<int32_t> w_output_pos;

        GGML_ASSERT(n_outputs <= n_outputs_max);

        w_output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch(); ++i) {
            // map an output id to a position in the batch
            int32_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT(pos < n_outputs);
                w_output_pos[pos] = i;
            }
        }

        io.write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            io.write(w_output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    // write logits
    {
        LLAMA_LOG_DEBUG("%s: - writing logits\n", __func__);

        const uint64_t logits_size = std::min((uint64_t) this->logits_size, (uint64_t) n_outputs * model.vocab.n_tokens());

        io.write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            io.write(logits, logits_size * sizeof(float));
        }
    }

    // write embeddings
    {
        LLAMA_LOG_DEBUG("%s: - writing embeddings\n", __func__);

        const uint64_t embd_size = std::min((uint64_t) this->embd_size, (uint64_t) n_outputs * model.hparams.n_embd);

        io.write(&embd_size, sizeof(embd_size));

        if (embd_size) {
            io.write(embd, embd_size * sizeof(float));
        }
    }

    LLAMA_LOG_DEBUG("%s: - writing KV self\n", __func__);
    kv_self->state_write(io);

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    // read output ids
    {
        LLAMA_LOG_DEBUG("%s: - reading output ids\n", __func__);

        auto n_outputs = this->n_outputs;
        io.read_to(&n_outputs, sizeof(n_outputs));

        if (n_outputs > output_reserve(n_outputs)) {
            throw std::runtime_error("could not reserve outputs");
        }

        std::vector<int32_t> output_pos;

        if (n_outputs) {
            output_pos.resize(n_outputs);
            io.read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= n_batch()) {
                    throw std::runtime_error(format("invalid output id, %d does not fit in batch size of %u", id, n_batch()));
                }
                this->output_ids[id] = i;
            }

            this->n_outputs = n_outputs;
        }
    }

    // read logits
    {
        LLAMA_LOG_DEBUG("%s: - reading logits\n", __func__);

        uint64_t logits_size;
        io.read_to(&logits_size, sizeof(logits_size));

        if (this->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            io.read_to(this->logits, logits_size * sizeof(float));
        }
    }

    // read embeddings
    {
        LLAMA_LOG_DEBUG("%s: - reading embeddings\n", __func__);

        uint64_t embd_size;
        io.read_to(&embd_size, sizeof(embd_size));

        if (this->embd_size < embd_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embd_size) {
            io.read_to(this->embd, embd_size * sizeof(float));
        }
    }

    LLAMA_LOG_DEBUG("%s: - reading KV self\n", __func__);
    kv_self->state_read(io);

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);

    kv_self->state_write(io, seq_id);

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);

    kv_self->state_read(io, seq_id);

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ 1.0f,
        /*.yarn_beta_fast              =*/ 32.0f,
        /*.yarn_beta_slow              =*/ 1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.logits_all                  =*/ false,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.flash_attn                  =*/ false,
        /*.no_perf                     =*/ true,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn = false;
    }

    if (params.flash_attn && model->hparams.n_embd_head_k != model->hparams.n_embd_head_v) {
        LLAMA_LOG_WARN("%s: flash_attn requires n_embd_head_k == n_embd_head_v - forcing off\n", __func__);
        params.flash_attn = false;
    }

    if (ggml_is_quantized(params.type_v) && !params.flash_attn) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

llama_kv_cache * llama_get_kv_self(llama_context * ctx) {
    return ctx->get_kv_self();
}

void llama_kv_self_update(llama_context * ctx) {
    ctx->kv_self_update();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_ith(i);
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

// llama adapter API

int32_t llama_set_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter,
            float scale) {
    ctx->set_adapter_lora(adapter, scale);

    return 0;
}

int32_t llama_rm_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter) {
    bool res = ctx->rm_adapter_lora(adapter);

    return res ? 0 : -1;
}

void llama_clear_adapter_lora(llama_context * ctx) {
    ctx->clear_adapter_lora();
}

int32_t llama_apply_adapter_cvec(
        llama_context * ctx,
                 const float * data,
                      size_t   len,
                     int32_t   n_embd,
                     int32_t   il_start,
                     int32_t   il_end) {
    bool res = ctx->apply_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// kv cache view
//

llama_kv_cache_view llama_kv_cache_view_init(const llama_context * ctx, int32_t n_seq_max) {
    const auto * kv = ctx->get_kv_self();
    if (kv == nullptr) {
        LLAMA_LOG_WARN("%s: the context does not have a KV cache\n", __func__);
        return {};
    }

    return llama_kv_cache_view_init(*kv, n_seq_max);
}

void llama_kv_cache_view_update(const llama_context * ctx, llama_kv_cache_view * view) {
    const auto * kv = ctx->get_kv_self();
    if (kv == nullptr) {
        LLAMA_LOG_WARN("%s: the context does not have a KV cache\n", __func__);
        return;
    }

    llama_kv_cache_view_update(view, kv);
}

//
// kv cache
//

// deprecated
int32_t llama_get_kv_cache_token_count(const llama_context * ctx) {
    return llama_kv_self_n_tokens(ctx);
}

int32_t llama_kv_self_n_tokens(const llama_context * ctx) {
    return llama_kv_cache_n_tokens(ctx->get_kv_self());
}

// deprecated
int32_t llama_get_kv_cache_used_cells(const llama_context * ctx) {
    return llama_kv_self_used_cells(ctx);
}

int32_t llama_kv_self_used_cells(const llama_context * ctx) {
    return llama_kv_cache_used_cells(ctx->get_kv_self());
}

// deprecated
void llama_kv_cache_clear(llama_context * ctx) {
    llama_kv_self_clear(ctx);
}

void llama_kv_self_clear(llama_context * ctx) {
    llama_kv_cache_clear(ctx->get_kv_self());
}

// deprecated
bool llama_kv_cache_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_self_seq_rm(ctx, seq_id, p0, p1);
}

bool llama_kv_self_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_cache_seq_rm(ctx->get_kv_self(), seq_id, p0, p1);
}

// deprecated
void llama_kv_cache_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_self_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_self_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_cache_seq_cp(ctx->get_kv_self(), seq_id_src, seq_id_dst, p0, p1);
}

// deprecated
void llama_kv_cache_seq_keep(
        llama_context * ctx,
         llama_seq_id   seq_id) {
    return llama_kv_self_seq_keep(ctx, seq_id);
}

void llama_kv_self_seq_keep(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_cache_seq_keep(ctx->get_kv_self(), seq_id);
}

// deprecated
void llama_kv_cache_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    return llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta);
}

void llama_kv_self_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    return llama_kv_cache_seq_add(ctx->get_kv_self(), seq_id, p0, p1, delta);
}

// deprecated
void llama_kv_cache_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    return llama_kv_self_seq_div(ctx, seq_id, p0, p1, d);
}

void llama_kv_self_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    return llama_kv_cache_seq_div(ctx->get_kv_self(), seq_id, p0, p1, d);
}

// deprecated
llama_pos llama_kv_cache_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_self_seq_pos_max(ctx, seq_id);
}

llama_pos llama_kv_self_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_cache_seq_pos_max(ctx->get_kv_self(), seq_id);
}

// deprecated
void llama_kv_cache_defrag(llama_context * ctx) {
    return llama_kv_self_defrag(ctx);
}

void llama_kv_self_defrag(llama_context * ctx) {
    llama_kv_cache_defrag(ctx->get_kv_self());
}

// deprecated
bool llama_kv_cache_can_shift(const llama_context * ctx) {
    return llama_kv_self_can_shift(ctx);
}

bool llama_kv_self_can_shift(const llama_context * ctx) {
    return llama_kv_cache_can_shift(ctx->get_kv_self());
}

// deprecated
void llama_kv_cache_update(llama_context * ctx) {
    llama_kv_self_update(ctx);
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return ctx->state_seq_get_size(seq_id);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//내가추가---------------------------------------------------------------------
const llama_kv_cache_unified *
llama_get_kv_cache_unified(const llama_context * ctx) {
    if (!ctx) return nullptr;
    return dynamic_cast<const llama_kv_cache_unified *>(ctx->kv_self.get());
}
//-------------------------------------------------------------------------
