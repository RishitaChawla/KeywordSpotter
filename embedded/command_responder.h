#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_COMMAND_RESPONDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_COMMAND_RESPONDER_H_

#include <stdint.h>

void RespondToCommand(int32_t current_time,
                      const char* found_command,
                      uint8_t score,
                      bool is_new_command);

#endif