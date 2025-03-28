// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_test_vectors.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(const embedlayernorm::OpData& data,
                    bool use_float16 = false,
                    bool sum_output = false,
                    bool broadcast_position_ids = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = DefaultRocmExecutionProvider().get() != nullptr;
  bool enable_dml = DefaultDmlExecutionProvider().get() != nullptr;
  bool enable_cpu = !use_float16;

  if (enable_cpu || enable_cuda || enable_dml || enable_rocm) {
    // Input and output shapes
    //   Input 0 - input_ids          : (batch_size, sequence_size)
    //   Input 1 - segment_ids        : (batch_size, sequence_size)
    //   Input 2 - word_embedding     : (,hidden_size)
    //   Input 3 - position_embedding : (,hidden_size)
    //   Input 4 - segment_embedding  : (,hidden_size)
    //   Input 5 - gamma              : (hidden_size)
    //   Input 6 - beta               : (hidden_size)
    //   Input 7 - mask               : (batch_size, sequence_size)
    //   Input 8 - position ids       : (batch_size, sequence_size) or (1, sequence_size)
    //   Output 0 - output            : (batch_size, sequence_size, hidden_size)
    //   Output 1 - mask_index        : (batch_size)
    //   Output 2 - embedding_sum     : (batch_size, sequence_size, hidden_size)

    std::vector<int64_t> input_ids_dims = {data.batch_size, data.sequence_size};
    std::vector<int64_t> segment_ids_dims = {data.batch_size, data.sequence_size};
    std::vector<int64_t> mask_dims = {data.batch_size, data.sequence_size};

    ASSERT_TRUE(data.word_embedding_data.size() % data.hidden_size == 0);
    std::vector<int64_t> word_embedding_dims = {
        static_cast<int64_t>(data.word_embedding_data.size() / data.hidden_size),
        data.hidden_size};

    ASSERT_TRUE(data.position_embedding_data.size() % data.hidden_size == 0);
    std::vector<int64_t> position_embedding_dims = {
        static_cast<int64_t>(data.position_embedding_data.size() / data.hidden_size),
        data.hidden_size};

    ASSERT_TRUE(data.segment_embedding_data.size() % data.hidden_size == 0);
    std::vector<int64_t> segment_embedding_dims = {
        static_cast<int64_t>(data.segment_embedding_data.size() / data.hidden_size),
        data.hidden_size};

    std::vector<int64_t> gamma_dims = {data.hidden_size};
    std::vector<int64_t> beta_dims = gamma_dims;
    std::vector<int64_t> output_dims = {data.batch_size, data.sequence_size, data.hidden_size};
    std::vector<int64_t> mask_index_dims = {data.batch_size};

    OpTester tester("EmbedLayerNormalization", 1, onnxruntime::kMSDomain);
    tester.AddInput<int32_t>("input_ids", input_ids_dims, data.input_ids_data);
    if (!data.has_segment) {
      tester.AddOptionalInputEdge<int32_t>();
    } else {
      tester.AddInput<int32_t>("segment_ids", segment_ids_dims, data.segment_ids_data);
    }
    if (use_float16) {
      tester.AddInput<MLFloat16>("word_embedding",
                                 word_embedding_dims,
                                 ToFloat16(data.word_embedding_data),
                                 /*is_initializer=*/true);
      tester.AddInput<MLFloat16>("position_embedding",
                                 position_embedding_dims,
                                 ToFloat16(data.position_embedding_data),
                                 /*is_initializer=*/true);
      if (!data.has_segment) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddInput<MLFloat16>("segment_embedding",
                                   segment_embedding_dims,
                                   ToFloat16(data.segment_embedding_data),
                                   /*is_initializer=*/true);
      }
      tester.AddInput<MLFloat16>("gamma",
                                 gamma_dims,
                                 ToFloat16(data.gamma_data),
                                 /*is_initializer=*/true);
      tester.AddInput<MLFloat16>("beta",
                                 beta_dims,
                                 ToFloat16(data.beta_data),
                                 /*is_initializer=*/true);
      tester.AddAttribute("epsilon", data.epsilon);
      if (data.has_mask && data.mask_data.size()) {
        tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
      }
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(data.output_data));
    } else {
      tester.AddInput<float>("word_embedding",
                             word_embedding_dims,
                             data.word_embedding_data,
                             /*is_initializer=*/true);
      tester.AddInput<float>("position_embedding",
                             position_embedding_dims,
                             data.position_embedding_data,
                             /*is_initializer=*/true);
      if (!data.has_segment) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddInput<float>("segment_embedding",
                               segment_embedding_dims,
                               data.segment_embedding_data,
                               /*is_initializer=*/true);
      }
      tester.AddInput<float>("gamma", gamma_dims, data.gamma_data, /*is_initializer=*/true);
      tester.AddInput<float>("beta", beta_dims, data.beta_data, /*is_initializer=*/true);
      tester.AddAttribute("epsilon", data.epsilon);
      if (data.has_mask && data.mask_data.size()) {
        tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
      }
      tester.AddOutput<float>("output", output_dims, data.output_data);
    }
    tester.AddAttribute("mask_index_type", static_cast<int64_t>(data.mask_index_type));
    if (data.mask_index_data.size()) {
      tester.AddOutput<int32_t>("mask_index", mask_index_dims, data.mask_index_data);
    } else {
      tester.AddOptionalOutputEdge<int32_t>();
    }
    if (sum_output) {
      std::vector<int64_t> embedding_sum_output_dims = output_dims;
      if (use_float16) {
        tester.AddOutput<MLFloat16>("embedding_sum", embedding_sum_output_dims, ToFloat16(data.embedding_sum_data));
      } else {
        tester.AddOutput<float>("embedding_sum", embedding_sum_output_dims, data.embedding_sum_data);
      }
    }
    if (data.position_ids_data.size() != 0) {
      if (broadcast_position_ids) {
        std::vector<int64_t> position_ids_dims = {1, data.sequence_size};
        tester.AddInput<int32_t>("position_ids", position_ids_dims, data.position_ids_data);
      } else {
        tester.AddInput<int32_t>("position_ids", input_ids_dims, data.position_ids_data);
      }
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    } else if (enable_rocm) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultRocmExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    } else if (enable_dml) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultDmlExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    } else {
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
    }
  }
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_Float16) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1(), /*use_float16=*/true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_PositionIds) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_PositionIds());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_PositionIdsDiffOrder) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_PositionIds(true));
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch3_PositionIds_BroadCast) {
  RunTest(embedlayernorm::EmbedLayerNormBatch3_PositionIds_BroadCast(),
          /*use_float16=*/false,
          /*sum_output=*/false,
          /*broadcast_position_ids=*/true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_EmbeddingSum) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_EmbeddingSum(), false, true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_EmbeddingSum_Float16) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_EmbeddingSum(), true, true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_EmbeddingSum_NoMaskIndex) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_EmbeddingSum_NoMaskIndex(),
          /* use_float16 = */ false,
          /* sum_output = */ true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch2) {
  RunTest(embedlayernorm::EmbedLayerNormBatch2());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch2_NoMask) {
  RunTest(embedlayernorm::EmbedLayerNormBatch2(/*has_mask=*/false));
}

// BatchSize > HiddenSize to reproduce mask processing bug
TEST(EmbedLayerNormTest, EmbedLayerNormLargeBatchSmallHiddenSize) {
  RunTest(embedlayernorm::EmbedLayerNormLargeBatchSmallHiddenSize());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch_Distill) {
  RunTest(embedlayernorm::EmbedLayerNormBatch_Distill());
}

}  // namespace test
}  // namespace onnxruntime
