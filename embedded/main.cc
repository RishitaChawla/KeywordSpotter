/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0
==============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "main_functions.h"

void tf_main(void) {
  setup();

  while (true) {
    loop();
  }
}

extern "C" void app_main() {
  ESP_LOGI("app_main", "Starting TensorFlow micro_speech project");

  xTaskCreate(
      (TaskFunction_t)&tf_main,
      "tensorflow",
      8 * 1024,
      NULL,
      8,
      NULL
  );

  vTaskDelete(NULL);
}