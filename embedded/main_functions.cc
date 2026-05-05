/* Copyright 2020-2023 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "main_functions.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Added for inference timing
#include "esp_timer.h" 

// Globals
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 200 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

void setup() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version mismatch!");
    return;
  }

  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddDepthwiseConv2D(); 

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed!");
    return;
  }

  model_input = interpreter->input(0);
  
  int input_total_elements = 1;
  for (int i = 0; i < model_input->dims->size; ++i) {
    input_total_elements *= model_input->dims->data[i];
  }

  if ((input_total_elements != kFeatureElementCount) || (model_input->type != kTfLiteInt8)) {
    MicroPrintf("Mismatch! Model wants total size %d, type %d", input_total_elements, model_input->type);
    return;
  }
  
  model_input_buffer = tflite::GetTensorData<int8_t>(model_input);

  static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  previous_time = 0;
  MicroPrintf("Setup complete. Listening...");
}

void loop() {
  if (interpreter == nullptr || feature_provider == nullptr || model_input_buffer == nullptr) {
    return; 
  }

  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  
  if (feature_status != kTfLiteOk || how_many_new_slices == 0) {
    return;
  }
  
  previous_time = current_time;

  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // --- START INFERENCE TIMING ---
  int64_t start_time = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  int64_t end_time = esp_timer_get_time();
  // --- END INFERENCE TIMING ---

  if (invoke_status != kTfLiteOk) return;

  TfLiteTensor* output = interpreter->output(0);
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  
  int max_idx = 0;
  float max_result = 0.0;

  // Print raw scores for ALL categories to find why Silence triggers Activate
  for (int i = 0; i < kCategoryCount; i++) {
    float current_result = (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) * output_scale;
    if (current_result > max_result) {
      max_result = current_result;
      max_idx = i;
    }
  }

  // Debug output: Only if someone is speaking or threshold is met
  if (max_result > 0.3f) {
    int64_t duration_ms = (end_time - start_time) / 1000;
    MicroPrintf("--- Detection ---");
    MicroPrintf("Inference Speed: %lld ms", duration_ms);
    
    // This loop prints every label's score so you can see the "competition"
    for (int i = 0; i < kCategoryCount; i++) {
        float score = (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) * output_scale;
        MicroPrintf("Label %d [%s]: %.2f", i, kCategoryLabels[i], static_cast<double>(score));
    }
    MicroPrintf("Winner: %s", kCategoryLabels[max_idx]);
  }
}