#include "command_responder.h"

#include <string.h>
#include <inttypes.h>

#include "esp_log.h"
#include "driver/gpio.h"

static const char* TAG = "command_responder";

/*
   ESP32-S3-EYE controllable LED:
   GPIO3

   GPIO3 setup:
   - Open-drain mode
   - Active LOW

   LED_ON  = 0
   LED_OFF = 1
*/
#define LED_GPIO GPIO_NUM_3

#define LED_ON  0
#define LED_OFF 1

static bool led_initialized = false;

static void init_led_once() {
    if (led_initialized) {
        return;
    }

    gpio_config_t io_conf = {};
    io_conf.pin_bit_mask = (1ULL << LED_GPIO);
    io_conf.mode = GPIO_MODE_OUTPUT_OD;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.intr_type = GPIO_INTR_DISABLE;

    gpio_config(&io_conf);

    // LED OFF every time board starts/restarts
    gpio_set_level(LED_GPIO, LED_OFF);

    led_initialized = true;

    ESP_LOGI(TAG, "GPIO3 LED initialized OFF");
}

void RespondToCommand(int32_t current_time,
                      const char* found_command,
                      uint8_t score,
                      bool is_new_command) {
    init_led_once();

    ESP_LOGI(TAG,
             "RespondToCommand called: command=%s, score=%u, time=%" PRId32 ", is_new=%d",
             found_command,
             score,
             current_time,
             is_new_command);

    if (is_new_command) {
        if (strcmp(found_command, "yes") == 0) {
            ESP_LOGI(TAG, "YES detected -> LED ON");
            gpio_set_level(LED_GPIO, LED_ON);
        }
        else if (strcmp(found_command, "no") == 0) {
            ESP_LOGI(TAG, "NO detected -> LED OFF");
            gpio_set_level(LED_GPIO, LED_OFF);
        }
        else {
            ESP_LOGI(TAG, "Other command detected: %s -> LED unchanged", found_command);
        }
    }
}