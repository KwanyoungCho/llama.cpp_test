// llama.cpp - 이 파일은 llama.cpp 라이브러리의 공개 API 구현을 담당하는 코어 소스 파일입니다.

// 내부 구현 헤더 포함
#include "llama-impl.h"

// 다양한 기능 모듈 헤더 파일들 포함
#include "llama-chat.h"     // 채팅 기능 관련 (채팅 템플릿, 메시지 포맷팅 등)
#include "llama-mmap.h"     // 메모리 매핑 기능 (대용량 모델 파일 효율적 로드)
#include "llama-vocab.h"    // 어휘 관리 (토큰화, 토큰 정보 등)
#include "llama-model-loader.h" // 모델 로딩 기능
#include "llama-model.h"    // 모델 구조체 및 관련 기능

// GGML 라이브러리 헤더 (행렬 연산 및 그래프 기반 계산을 위한 라이브러리)
#include "ggml.h"
#include "ggml-backend.h"   // 다양한 하드웨어 백엔드 지원 (CPU, GPU 등)

// 표준 라이브러리 헤더
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

// MSVC 컴파일러 사용 시 경고 비활성화
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// 인터페이스 구현 부분
//

// 샘플러 체인 기본 파라미터 설정 함수
// 샘플러 체인은 여러 샘플링 방법(top-k, top-p 등)을 조합할 수 있게 해줌
struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true, // 성능 측정 비활성화
    };

    return result;
}

// 최대 지원 디바이스 수 반환 (GPU 등)
size_t llama_max_devices(void) {
    return 16; // 최대 16개 디바이스 지원
}

// 메모리 매핑(mmap) 지원 여부 확인
// mmap은 파일을 메모리에 직접 매핑하여 I/O 성능 향상
bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED; // 플랫폼에 따른 mmap 지원 여부 반환
}

// 메모리 락(mlock) 지원 여부 확인
// mlock은 메모리가 스왑되지 않도록 보장하여 성능 향상
bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED; // 플랫폼에 따른 mlock 지원 여부 반환
}

// GPU 오프로딩 지원 여부 확인
// 모델의 일부를 GPU에서 실행 가능한지 확인
bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           llama_supports_rpc(); // GPU 백엔드 또는 RPC 지원 확인
}

// 원격 프로시저 호출(RPC) 지원 여부 확인
// 모델 실행을 원격 서버에 위임 가능
bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr; // RPC 백엔드 등록 여부 확인
}

// 백엔드 초기화 함수
// GGML 라이브러리 및 관련 리소스 초기화
void llama_backend_init(void) {
    ggml_time_init(); // 시간 측정 기능 초기화

    // f16(반정밀도) 변환 테이블 초기화
    // 임시 컨텍스트를 생성하고 해제하여 GGML 내부 자료구조 초기화
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

// NUMA(Non-Uniform Memory Access) 초기화 함수
// 다중 CPU 시스템에서 메모리 액세스 최적화
void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        numa_init_fn(numa); // 지정된 NUMA 전략으로 CPU 백엔드 초기화
    }
}

// 백엔드 리소스 해제 함수
void llama_backend_free(void) {
    ggml_quantize_free(); // 양자화 관련 리소스 해제
}

// 마이크로초 단위 시간 측정 함수
int64_t llama_time_us(void) {
    return ggml_time_us(); // GGML의 시간 측정 함수 사용
}

// 모델 로드 함수 (내부용)
// 반환값: 0=성공, -1=오류, -2=사용자에 의한 취소
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    // 로딩 시간 측정 - mmap의 페이지 폴트 등을 고려하여 첫 평가 후 다시 계산됨
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        // 모델 로더 초기화 (파일 경로, 분할 파일 목록, mmap 사용 여부 등 설정)
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides);

        ml.print_info(); // 모델 정보 출력

        model.hparams.vocab_only = params.vocab_only; // 어휘만 로드할지 여부 설정

        try {
            model.load_arch(ml); // 모델 아키텍처 로드 (네트워크 구조)
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml); // 모델 하이퍼파라미터 로드 (은닉층 크기, 층 수 등)
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml); // 모델 어휘 로드 (토큰화, 토큰 정보 등)
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml); // 모델 통계 정보 로드
        model.print_info(); // 최종 모델 정보 출력

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0; // 어휘만 로드하는 경우 텐서 로드 건너뜀
        }

        if (!model.load_tensors(ml)) { // 모델 가중치 텐서 로드
            return -2; // 사용자에 의한 취소 (콜백에서 중단 신호)
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1; // 오류 발생
    }

    return 0; // 성공
}

// 파일에서 모델 로드 구현 (여러 API 함수의 공통 구현)
static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    ggml_time_init(); // 시간 측정 초기화

    // 진행 상황 콜백 설정 (사용자 지정 콜백이 없는 경우 기본 콜백 제공)
    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT("."); // 진행 상태를 점으로 표시
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true; // 계속 진행
        };
    }

    // 모델 객체 생성
    llama_model * model = new llama_model(params);

    // 모델과 함께 사용할 디바이스 목록 생성
    if (params.devices) {
        // 사용자가 지정한 디바이스 목록 사용
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        // 사용 가능한 모든 디바이스 자동 감지
        std::vector<ggml_backend_dev_t> rpc_servers;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // CPU 백엔드는 별도로 처리되므로 건너뜀
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU:
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                        rpc_servers.push_back(dev); // RPC 서버는 별도 관리
                    } else {
                        model->devices.push_back(dev); // GPU 디바이스 추가
                    }
                    break;
            }
        }
        // RPC 서버를 목록 앞쪽에 추가 (우선 활용)
        if (!rpc_servers.empty()) {
            model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());
        }
    }

    // 단일 GPU 모드 사용 시 지정된 메인 GPU만 남기고 나머지 제거
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0 || params.main_gpu >= (int)model->devices.size()) {
            LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %d)\n", __func__, params.main_gpu, (int)model->devices.size());
            llama_model_free(model);
            return nullptr;
        }
        ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
        model->devices.clear();
        model->devices.push_back(main_gpu);
    }

    // 사용 가능한 디바이스 정보 출력
    for (auto * dev : model->devices) {
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        LLAMA_LOG_INFO("%s: using device %s (%s) - %zu MiB free\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), free/1024/1024);
    }

    // 모델 로드 실행
    const int status = llama_model_load(path_model, splits, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model; // 성공적으로 로드된 모델 반환
}

// 이전 버전 호환을 위한 함수 (deprecated)
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

// 파일에서 모델 로드 (단일 모델 파일)
struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {}; // 빈 분할 목록 (단일 파일)
    return llama_model_load_from_file_impl(path_model, splits, params);
}

// 여러 분할 파일에서 모델 로드
struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]); // 분할 파일 경로 추가
    }
    return llama_model_load_from_file_impl(splits.front(), splits, params);
}

//
// 채팅 템플릿 관련 기능
//

// 채팅 메시지에 템플릿 적용 함수
// 다양한 모델이 요구하는 채팅 포맷(ChatML, Llama2 등)에 맞게 메시지 포맷팅
int32_t llama_chat_apply_template(
                              const char * tmpl, // 템플릿 이름 (chatml, llama2 등)
         const struct llama_chat_message * chat, // 채팅 메시지 배열
                                  size_t   n_msg, // 메시지 수
                                    bool   add_ass, // 비어있는 응답 추가 여부
                                    char * buf, // 결과를 저장할 버퍼
                                 int32_t   length) { // 버퍼 길이
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl); // 기본값: chatml

    // 채팅 메시지를 벡터로 변환
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    // 템플릿 감지 및 적용
    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1; // 알 수 없는 템플릿
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res; // 오류 발생
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length); // 결과를 버퍼에 복사
    }
    return res; // 성공적으로 포맷팅된 문자 수 반환
}

//
// 모델 분할 관련 기능
//

// 분할 경로 생성 함수 (대용량 모델을 여러 파일로 분할한 경우)
int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

// 분할 접두사 추출 함수 (분할 파일 경로에서 공통 접두사 추출)
int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // 접미사 패턴 확인 및 접두사 추출
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

// 시스템 정보 출력 함수 (사용 가능한 백엔드 및 기능 정보)
const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // 정적 문자열이므로 이전 호출 데이터 제거

    // 등록된 모든 백엔드의 기능 정보 수집
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str(); // 수집된 정보 문자열 반환
}
